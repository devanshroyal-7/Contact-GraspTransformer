import os
import torch
import argparse
from torch.utils.data import DataLoader
from data.dataset import CGNDataset
from models.model import ContactGraspNet
from loss import CGNLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/out', help='Path to datasets')
    parser.add_argument('--backbone', type=str, default='pn2', choices=['pn2', 'ptv3'], help='Backbone type')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--overfit_one_batch', action='store_true', help='Test flag')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {args.backbone} backbone on {device}")
    
    model = ContactGraspNet(backbone_type=args.backbone).to(device)
    criterion = CGNLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if os.path.exists(args.data_dir) and len(os.listdir(args.data_dir)) > 0:
        dataset = CGNDataset(args.data_dir)
        # Using a small mock loader
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print("No real data found. Using mock random data to verify pipeline builds.")
        mock_data = [{
            'points': torch.randn(2048, 3),
            'confidence': torch.randint(0, 2, (2048,)).float(),
            'approach_dirs': torch.randn(2048, 3),
            'base_dirs': torch.randn(2048, 3),
            'widths': torch.rand(2048) * 0.1
        } for _ in range(8)]
        train_loader = DataLoader(mock_data, batch_size=args.batch_size)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            points = batch['points'].to(device)
            # targets
            targets = {k: v.to(device) for k, v in batch.items() if k != 'points'}
            
            optimizer.zero_grad()
            preds = model(points)
            
            loss_dict = criterion(preds, targets)
            loss = loss_dict['loss']
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if args.overfit_one_batch:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Conf: {loss_dict['l_conf'].item():.4f}")
                break
                
        if not args.overfit_one_batch:
            print(f"Epoch {epoch} | Avg Loss: {total_loss / len(train_loader):.4f}")
            
    print("Training finished.")

if __name__ == '__main__':
    main()
