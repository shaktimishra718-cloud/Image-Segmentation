from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

print (f"Total number of batches in trainloader : {len(trainloader)}")
print (f"Total number of batches in validloader : {len(validloader)}")

for image, mask in trainloader:
  
  print(f"Image batch shape : {image.shape}")
  print(f"Mask batch shape : {mask.shape}")
  break
