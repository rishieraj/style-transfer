# Training loop
plot_losses = []

for epoch in range(0, num_epochs):
    running_total_loss = 0.0
    running_mse_loss = 0.0
    running_style_loss = 0.0

    # Scale the losses dynamically
    lamda_n = 1.0
    lamda_c = min(1.0, (epoch+1)/num_epochs)
    lamda_s = min(1.0, (epoch+1)/num_epochs)

    for step, batch in enumerate(train_dataloader):

        content_images = batch['pixel_values'].to(device)
        labels = batch['input_ids'].to(device)

        # Extracting style embeddings from MLP encoder
        # style_feature_stat = style_feature_stat.clone().detach().requires_grad_(True)
        # style_feature_stat = style_feature_stat.to(device)
        style_embeddings = style_encoder(style_feature_stat)
        # print(f"style_embeddings.requires_grad: {style_embeddings.requires_grad}")

        for name, module in unet.named_modules():
            if hasattr(module, "explicit_adaptation"):
                module.style_embeddings = style_embeddings

        # The random starting point
        latents = vae.encode(content_images.to(dtype=torch.float32)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that is added to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = get_text_embeddings(labels)

        # # Get the prediction
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # MSE loss for training the adaptor
        mse_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Reset predicted noise for multi-step DDIM denoising
        noise_pred = 0.0
        # Setting scheduler timestep
        noise_scheduler.set_timesteps(diffusion_steps)
        for i, t in enumerate(noise_scheduler.timesteps):
            noise_pred = noise_pred + unet(noisy_latents, t, encoder_hidden_states).sample
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents, eta=0.0, use_clipped_model_output=True).prev_sample

        # Decode the intermediate images
        with torch.no_grad():
            pseudo_x0 = 1.0 / vae.config.scaling_factor * noisy_latents
            decoded_images = vae.decode(pseudo_x0).sample
            decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)

        # # Display the decoded images and timesteps
        # display(TF.to_pil_image(decoded_images[0]))
        # print(timesteps[0])

        # --- Compute Additional Losses ---
        # # (A) Content Loss: using a perceptual/CLIP-based measure to compare decoded images and original content.
        # content_loss = get_content_loss(decoded_images, content_images, device)

        # (B) Style Loss: compare Gram matrices of features extracted from patches of the decoded image and the reference style image.
        style_image_expanded = style_image.expand(batch_size, -1, -1, -1)
        style_patches = get_image_patches(style_image_expanded, num_patches, patch_size, device)
        target_patches = get_image_patches(decoded_images, num_patches, patch_size, device)
        patch_style_loss = get_style_loss(style_patches, target_patches)

        # Calculate total loss
        total_loss = lamda_n * mse_loss + lamda_s * patch_style_loss #+ lamda_c * content_loss

        running_total_loss += total_loss.item()
        running_mse_loss += mse_loss.item()
        running_style_loss += patch_style_loss.item()

        # for name, param in unet.named_parameters():  # 'model' can be unet, style_encoder, or your combined trainable parameters
        #     if param.requires_grad and param.grad is not None:
        #         grad_flat = param.grad.flatten()
        #         norm = grad_flat.norm().item()
        #         grad_min = grad_flat.min().item()
        #         grad_max = grad_flat.max().item()
        #         grad_mean = grad_flat.mean().item()
        #         grad_std = grad_flat.std().item()
        #         print(f"{name} -- norm: {norm:.6f}, min: {grad_min:.6f}, max: {grad_max:.6f}, mean: {grad_mean:.6f}, std: {grad_std:.6f}")

        # for name, param in style_encoder.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         grad_flat = param.grad.flatten()
        #         norm = grad_flat.norm().item()
        #         grad_min = grad_flat.min().item()
        #         grad_max = grad_flat.max().item()
        #         grad_mean = grad_flat.mean().item()
        #         grad_std = grad_flat.std().item()
        #         print(f"{name} -- norm: {norm:.6f}, min: {grad_min:.6f}, max: {grad_max:.6f}, mean: {grad_mean:.6f}, std: {grad_std:.6f}")

        optimizer.zero_grad()
        total_loss.backward()
        # # Gradient clipping:
        # torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        lr_scheduler.step()

    avg_total_loss = running_total_loss / len(train_dataloader)
    avg_mse_loss = running_mse_loss / len(train_dataloader)
    avg_style_loss = running_style_loss / len(train_dataloader)

    if (epoch+1) % 1 == 0:
        plot_losses.append(avg_total_loss)
        print(f"Epoch {epoch+1}: Total Loss: {avg_total_loss:.6f} | MSE: {avg_mse_loss:.6f} | Style: {avg_style_loss:.6f}")