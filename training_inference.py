optimiser = torch.optim.Adam(model.parameters(), lr=LR)

best_valid_loss = np.inf

for epoch in range(EPOCHS):
  train_loss = train_fn(trainloader, model, optimiser)
  valid_loss = eval_fn(validloader, model)
  
  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    print("Saved Best Model!")
    best_valid_loss = valid_loss
    
  print(f"Epoch : {epoch+1} Train Loss : {train_loss} Valid Loss : {valid_loss}")

idx = 2

model.load_state_dict(torch.load('/content/best_model.pt'))
image, mask = validset[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0))#(c , h , w) -> (1 , c , h , w [1 is batch dimension])

pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5) * 1.0

helper.show_image(image, mask, pred_mask.detach().cpu().squeeze(0))


