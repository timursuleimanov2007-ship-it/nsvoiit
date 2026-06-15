import os
import json
import glob
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.utils import make_grid
import random
import time
from datetime import datetime

def user():
    base_path = os.path.dirname(__file__)
    user_folder = os.path.join(base_path, "User")
    
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
        print(f"Создана папка: {user_folder}")
        print("Положите туда иконки и запустите снова")
        return [], []
    
    supported_extensions = {'.png', '.jpg', '.jpeg', '.ico', '.svg', '.icns', '.gif', '.bmp', '.tiff', '.webp'}
    
    converted_folder = os.path.join(user_folder, "converted")
    os.makedirs(converted_folder, exist_ok=True)
    
    image_files = []
    
    for filename in os.listdir(user_folder):
        file_path = os.path.join(user_folder, filename)
        
        if os.path.isdir(file_path):
            continue
        
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue
        
        image_files.append(file_path)
    
    if not image_files:
        print("В папке User не найдено изображений")
        print(f"Поддерживаются: {supported_extensions}")
        return [], []
    
    print(f"Найдено изображений: {len(image_files)}")
    
    images = []
    image_paths = []
    
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                target_size = 128
                img_w, img_h = img.size
                scale = min(target_size / img_w, target_size / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                
                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
                x_offset = (target_size - new_w) // 2
                y_offset = (target_size - new_h) // 2
                new_img.paste(img_resized, (x_offset, y_offset))
                
                name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
                output_filename = f"{name_without_ext}_converted.png"
                output_path = os.path.join(converted_folder, output_filename)
                new_img.save(output_path, "PNG")
                
                images.append(new_img)
                image_paths.append(output_path)
                
                print(f"{os.path.basename(img_path)} → {output_filename}")
                
        except Exception as e:
            print(f"Ошибка {os.path.basename(img_path)}: {e}")
    
    print(f"\nКонвертировано {len(images)} иконок в {converted_folder}")
    
    return images, image_paths

def loadDirection(labels_file, root_dir, batch_size=32, train_ratio=0.8, image_size=128):
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    direction_to_idx = {direction: i for i, direction in enumerate(sorted(set(item["direction"] for item in labels)))}
    idx_to_direction = {i: direction for direction, i in direction_to_idx.items()}
    num_directions = len(direction_to_idx)
    
    print(f"Загружено иконок: {len(labels)}")
    print(f"Направления дизайна: {list(direction_to_idx.keys())}")
    
    direction_groups = {}
    for item in labels:
        direction = item["direction"]
        if direction not in direction_groups:
            direction_groups[direction] = []
        direction_groups[direction].append(item)
    
    print(f"\nРаспределение по направлениям:")
    for direction, items in direction_groups.items():
        print(f"  {direction}: {len(items)} иконок")
    
    max_count = max(len(items) for items in direction_groups.values())
    target_count = max_count * 2
    print(f"\nМаксимальное количество иконок в одном направлении: {max_count}")
    print(f"Целевое количество для каждого направления (x2): {target_count}")
    
    def get_heavy_augmentation():
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_light_augmentation():
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    balanced_items = []
    
    for direction, items in direction_groups.items():
        current_count = len(items)
        if current_count < target_count:
            copies_needed = target_count - current_count
            copies_per_item = copies_needed // current_count + 1
            
            print(f"\n{direction}: {current_count} -> {target_count} (+{copies_needed} аугментаций)")
            
            for item in items:
                balanced_items.append((item, False))
                for _ in range(copies_per_item):
                    balanced_items.append((item, True))
        else:
            print(f"\n{direction}: {current_count} -> {target_count} (без изменений)")
            for item in items:
                balanced_items.append((item, False))
    
    random.shuffle(balanced_items)
    
    def get_item(item_data, is_val=False):
        item, is_augmented = item_data
        img_path = os.path.join(root_dir, item["file"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        
        if is_val:
            transform = val_transform
        elif is_augmented:
            transform = get_heavy_augmentation()
        else:
            transform = get_light_augmentation()
        
        image = transform(image)
        direction_label = direction_to_idx[item["direction"]]
        
        return image, direction_label
    
    train_size = int(train_ratio * len(balanced_items))
    val_size = len(balanced_items) - train_size
    
    train_items = balanced_items[:train_size]
    val_items = balanced_items[train_size:]
    
    train_data = [get_item(item, is_val=False) for item in train_items]
    val_data = [get_item(item, is_val=True) for item in val_items]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nTrain (направления): {len(train_data)} иконок")
    print(f"Val (направления): {len(val_data)} иконок")
    
    train_labels = [label for _, label in train_data]
    label_counts = {}
    for s in train_labels:
        name = idx_to_direction[s]
        label_counts[name] = label_counts.get(name, 0) + 1
    
    print(f"\nРаспределение по направлениям в Train (после балансировки):")
    for name, count in sorted(label_counts.items()):
        print(f"  {name}: {count}")
    
    return train_loader, val_loader, num_directions, idx_to_direction

def loadStyle(labels_file, root_dir, batch_size=32, train_ratio=0.8, image_size=128):
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    style_to_idx = {style: i for i, style in enumerate(sorted(set(item["style"] for item in labels)))}
    idx_to_style = {i: style for style, i in style_to_idx.items()}
    num_styles = len(style_to_idx)
    
    print(f"Загружено иконок: {len(labels)}")
    print(f"Стили: {list(style_to_idx.keys())}")
    
    style_groups = {}
    for item in labels:
        style = item["style"]
        if style not in style_groups:
            style_groups[style] = []
        style_groups[style].append(item)
    
    print(f"\nРаспределение по стилям:")
    for style, items in style_groups.items():
        print(f"  {style}: {len(items)} иконок")
    
    max_count = max(len(items) for items in style_groups.values())
    target_count = max_count * 2
    print(f"\nМаксимальное количество иконок в одном стиле: {max_count}")
    print(f"Целевое количество для каждого стиля (x2): {target_count}")
    
    def get_heavy_augmentation():
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_light_augmentation():
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    balanced_items = []
    
    for style, items in style_groups.items():
        current_count = len(items)
        if current_count < target_count:
            copies_needed = target_count - current_count
            copies_per_item = copies_needed // current_count + 1
            
            print(f"\n{style}: {current_count} -> {target_count} (+{copies_needed} аугментаций)")
            
            for item in items:
                balanced_items.append((item, False))
                for _ in range(copies_per_item):
                    balanced_items.append((item, True))
        else:
            print(f"\n{style}: {current_count} -> {target_count} (без изменений)")
            for item in items:
                balanced_items.append((item, False))
    
    random.shuffle(balanced_items)
    
    def get_item(item_data, is_val=False):
        item, is_augmented = item_data
        img_path = os.path.join(root_dir, item["file"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        
        if is_val:
            transform = val_transform
        elif is_augmented:
            transform = get_heavy_augmentation()
        else:
            transform = get_light_augmentation()
        
        image = transform(image)
        style_label = style_to_idx[item["style"]]
        
        return image, style_label
    
    train_size = int(train_ratio * len(balanced_items))
    val_size = len(balanced_items) - train_size
    
    train_items = balanced_items[:train_size]
    val_items = balanced_items[train_size:]
    
    train_data = [get_item(item, is_val=False) for item in train_items]
    val_data = [get_item(item, is_val=True) for item in val_items]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nTrain (стили): {len(train_data)} иконок")
    print(f"Val (стили): {len(val_data)} иконок")
    
    train_labels = [label for _, label in train_data]
    label_counts = {}
    for s in train_labels:
        name = idx_to_style[s]
        label_counts[name] = label_counts.get(name, 0) + 1
    
    print(f"\nРаспределение по стилям в Train (после балансировки):")
    for name, count in sorted(label_counts.items()):
        print(f"  {name}: {count}")
    
    return train_loader, val_loader, num_styles, idx_to_style

def loadOS(labels_file, root_dir, batch_size=32, train_ratio=0.8, image_size=128):
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    os_to_idx = {os_name: i for i, os_name in enumerate(sorted(set(item["os"] for item in labels)))}
    idx_to_os = {i: os_name for os_name, i in os_to_idx.items()}
    num_os = len(os_to_idx)
    
    print(f"Загружено иконок: {len(labels)}")
    print(f"ОС: {list(os_to_idx.keys())}")
    
    os_groups = {}
    for item in labels:
        os_name = item["os"]
        if os_name not in os_groups:
            os_groups[os_name] = []
        os_groups[os_name].append(item)
    
    print(f"\nРаспределение по ОС:")
    for os_name, items in os_groups.items():
        print(f"  {os_name}: {len(items)} иконок")
    
    max_count = max(len(items) for items in os_groups.values())
    target_count = max_count * 2
    print(f"\nМаксимальное количество иконок в одной ОС: {max_count}")
    print(f"Целевое количество для каждой ОС (x2): {target_count}")
    
    def get_heavy_augmentation():
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_light_augmentation():
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    balanced_items = []
    
    for os_name, items in os_groups.items():
        current_count = len(items)
        if current_count < target_count:
            copies_needed = target_count - current_count
            copies_per_item = copies_needed // current_count + 1
            
            print(f"\n{os_name}: {current_count} -> {target_count} (+{copies_needed} аугментаций)")
            
            for item in items:
                balanced_items.append((item, False))
                for _ in range(copies_per_item):
                    balanced_items.append((item, True))
        else:
            print(f"\n{os_name}: {current_count} -> {target_count} (без изменений)")
            for item in items:
                balanced_items.append((item, False))
    
    random.shuffle(balanced_items)
    
    def get_item(item_data, is_val=False):
        item, is_augmented = item_data
        img_path = os.path.join(root_dir, item["file"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        
        if is_val:
            transform = val_transform
        elif is_augmented:
            transform = get_heavy_augmentation()
        else:
            transform = get_light_augmentation()
        
        image = transform(image)
        os_label = os_to_idx[item["os"]]
        
        return image, os_label
    
    train_size = int(train_ratio * len(balanced_items))
    val_size = len(balanced_items) - train_size
    
    train_items = balanced_items[:train_size]
    val_items = balanced_items[train_size:]
    
    train_data = [get_item(item, is_val=False) for item in train_items]
    val_data = [get_item(item, is_val=True) for item in val_items]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nTrain (ОС): {len(train_data)} иконок")
    print(f"Val (ОС): {len(val_data)} иконок")
    
    train_labels = [label for _, label in train_data]
    label_counts = {}
    for s in train_labels:
        name = idx_to_os[s]
        label_counts[name] = label_counts.get(name, 0) + 1
    
    print(f"\nРаспределение по ОС в Train (после балансировки):")
    for name, count in sorted(label_counts.items()):
        print(f"  {name}: {count}")
    
    return train_loader, val_loader, num_os, idx_to_os

def findDirection():
    base_path = os.path.dirname(__file__)
    DATASET_PATH = os.path.join(base_path, "НСвОИиТ курсовая датасет_converted")
    LABELS_FILE = os.path.join(DATASET_PATH, "labels_remapped.json")
    MODEL_PATH = os.path.join(base_path, "direction_model_best.pth")
    
    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nУстройство: {DEVICE}")
    
    if os.path.exists(MODEL_PATH):
        print(f"\nНайдена сохранённая модель: {MODEL_PATH}")
        answer = input("Загрузить сохранённую модель? (y/n): ")
        if answer.lower() == 'y':
            train_loader, val_loader, num_directions, idx_to_direction = loadDirection(
                labels_file=LABELS_FILE,
                root_dir=DATASET_PATH,
                batch_size=BATCH_SIZE,
                train_ratio=0.8,
                image_size=IMAGE_SIZE
            )
            
            direction_names = list(idx_to_direction.values())
            
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_directions)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = model.to(DEVICE)
            
            print(f"Модель загружена: {MODEL_PATH}")
            
            def predict(image):
                model.eval()
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    probs = {direction_names[i]: probabilities[0][i].item() for i in range(num_directions)}
                return probs
            return predict
    
    train_loader, val_loader, num_directions, idx_to_direction = loadDirection(
        labels_file=LABELS_FILE,
        root_dir=DATASET_PATH,
        batch_size=BATCH_SIZE,
        train_ratio=0.8,
        image_size=IMAGE_SIZE
    )
    
    all_val_directions = []
    for _, direction_labels in val_loader:
        all_val_directions.extend(direction_labels.numpy())
    
    unique_directions = sorted(set(all_val_directions))
    direction_names = [idx_to_direction[i] for i in unique_directions]
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_val_directions, bins=len(unique_directions), alpha=0.7, color='purple', edgecolor='black')
    plt.xticks(unique_directions, direction_names, rotation=45)
    plt.title("Распределение направлений дизайна в валидационной выборке")
    plt.xlabel("Направление")
    plt.ylabel("Количество иконок")
    plt.tight_layout()
    plt.savefig("direction_distribution.png")
    plt.close()
    print("Гистограмма направлений сохранена: direction_distribution.png")
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_directions)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    print("\nОбучение модели для определения направления дизайна...")
    
    train_losses = []
    val_accuracies = []
    best_acc = 0
    best_epoch = 0
    best_model_state = None
    patience = 7
    patience_counter = 0
    all_predictions = []
    all_labels = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, direction_labels in train_loader:
            images = images.to(DEVICE)
            direction_labels = direction_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, direction_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        correct = 0
        total = 0
        epoch_predictions = []
        epoch_labels = []
        
        with torch.no_grad():
            for images, direction_labels in val_loader:
                images = images.to(DEVICE)
                direction_labels = direction_labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += direction_labels.size(0)
                correct += (predicted == direction_labels).sum().item()
                
                epoch_predictions.extend(predicted.cpu().numpy())
                epoch_labels.extend(direction_labels.cpu().numpy())
        
        acc = 100 * correct / total
        val_accuracies.append(acc)
        
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            all_predictions = epoch_predictions
            all_labels = epoch_labels
            patience_counter = 0
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% – Новый лучший")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% (лучший: {best_acc:.2f}%)")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping на эпохе {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nЗагружена лучшая модель с эпохи {best_epoch}, точность {best_acc:.2f}%")
    
    print(f"\nФинальная точность: {best_acc:.2f}%")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', linewidth=2, markersize=4)
    plt.title("График потерь при обучении (определение направления дизайна)")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("direction_loss.png")
    plt.close()
    print("График потерь сохранён: direction_loss.png")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', linewidth=2, markersize=4, color='green')
    plt.axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.2f}%')
    plt.title("График точности на валидации (определение направления дизайна)")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("direction_accuracy.png")
    plt.close()
    print("График точности сохранён: direction_accuracy.png")
    
    class_names = direction_names
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Confusion Matrix (Направления дизайна) - Лучшая точность: {best_acc:.2f}%")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), 
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.xlabel("Предсказано")
    plt.ylabel("Истина")
    plt.tight_layout()
    plt.savefig("direction_confusion_matrix.png")
    plt.close()
    print("Confusion matrix сохранена: direction_confusion_matrix.png")
    
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print("\nОтчёт классификации (Направления дизайна):")
    print(report)
    
    with open("direction_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Отчёт сохранён: direction_report.txt")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")
    
    def predict(image):
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            probs = {direction_names[i]: probabilities[0][i].item() for i in range(num_directions)}
        return probs
    
    return predict

def findStyle():
    base_path = os.path.dirname(__file__)
    DATASET_PATH = os.path.join(base_path, "НСвОИиТ курсовая датасет_converted")
    LABELS_FILE = os.path.join(DATASET_PATH, "labels_remapped.json")
    MODEL_PATH = os.path.join(base_path, "style_model_best.pth")
    
    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Словарь: направление → список стилей
    DIRECTION_TO_STYLES = {
        "Skeuomorphism": ["Apple Skeuomorphism", "Windows Aero"],
        "Flat": ["Apple Flat by Jony Ive", "Windows Fluent", "Windows Metro"],
        "Glassmorphism": ["Apple Liquid Glass"],
        "Retro": ["Windows 80-90s Retro"]
    }
    
    # Обратный словарь: стиль → направление
    STYLE_TO_DIRECTION = {}
    for direction, styles in DIRECTION_TO_STYLES.items():
        for style in styles:
            STYLE_TO_DIRECTION[style] = direction
    
    print(f"\nУстройство: {DEVICE}")
    
    if os.path.exists(MODEL_PATH):
        print(f"\nНайдена сохранённая модель: {MODEL_PATH}")
        answer = input("Загрузить сохранённую модель? (y/n): ")
        if answer.lower() == 'y':
            train_loader, val_loader, num_styles, idx_to_style = loadStyle(
                labels_file=LABELS_FILE,
                root_dir=DATASET_PATH,
                batch_size=BATCH_SIZE,
                train_ratio=0.8,
                image_size=IMAGE_SIZE
            )
            
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_styles)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = model.to(DEVICE)
            
            print(f"Модель загружена: {MODEL_PATH}")
            
            style_names = list(idx_to_style.values())
            
            def predict(image, direction=None):
                model.eval()
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    probs = {style_names[i]: probabilities[0][i].item() for i in range(num_styles)}
                
                # Если передано направление, фильтруем стили
                if direction is not None:
                    allowed_styles = DIRECTION_TO_STYLES.get(direction, [])
                    if allowed_styles:
                        probs = {s: p for s, p in probs.items() if s in allowed_styles}
                        # Нормализуем вероятности
                        total = sum(probs.values())
                        if total > 0:
                            probs = {s: p / total for s, p in probs.items()}
                
                return probs
            return predict
    
    train_loader, val_loader, num_styles, idx_to_style = loadStyle(
        labels_file=LABELS_FILE,
        root_dir=DATASET_PATH,
        batch_size=BATCH_SIZE,
        train_ratio=0.8,
        image_size=IMAGE_SIZE
    )
    
    all_val_styles = []
    for _, style_labels in val_loader:
        all_val_styles.extend(style_labels.numpy())
    
    unique_styles = sorted(set(all_val_styles))
    style_names_list = [idx_to_style[i] for i in unique_styles]
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_val_styles, bins=len(unique_styles), alpha=0.7, color='green', edgecolor='black')
    plt.xticks(unique_styles, style_names_list, rotation=45, fontsize=10)
    plt.title("Распределение стилей в валидационной выборке")
    plt.xlabel("Стиль")
    plt.ylabel("Количество иконок")
    plt.tight_layout()
    plt.savefig("style_distribution.png")
    plt.close()
    print("Гистограмма стилей сохранена: style_distribution.png")
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_styles)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    print("\nОбучение модели для определения стиля...")
    print(f"Количество классов стилей: {num_styles}")
    
    train_losses = []
    val_accuracies = []
    best_acc = 0
    best_epoch = 0
    best_model_state = None
    patience = 5
    patience_counter = 0
    all_predictions = []
    all_labels = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, style_labels in train_loader:
            images = images.to(DEVICE)
            style_labels = style_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, style_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        correct = 0
        total = 0
        epoch_predictions = []
        epoch_labels = []
        
        with torch.no_grad():
            for images, style_labels in val_loader:
                images = images.to(DEVICE)
                style_labels = style_labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += style_labels.size(0)
                correct += (predicted == style_labels).sum().item()
                
                epoch_predictions.extend(predicted.cpu().numpy())
                epoch_labels.extend(style_labels.cpu().numpy())
        
        acc = 100 * correct / total
        val_accuracies.append(acc)
        
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            all_predictions = epoch_predictions
            all_labels = epoch_labels
            patience_counter = 0
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% – НОВЫЙ ЛУЧШИЙ")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% (лучший: {best_acc:.2f}%)")
        
        if patience_counter >= patience:
            print(f"\nРанняя остановка на эпохе {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nЗагружена лучшая модель с эпохи {best_epoch}, точность {best_acc:.2f}%")
    
    print(f"\nФинальная точность: {best_acc:.2f}%")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', linewidth=2, markersize=4)
    plt.title("График потерь при обучении (определение стиля)")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("style_loss.png")
    plt.close()
    print("График потерь сохранён: style_loss.png")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', linewidth=2, markersize=4, color='green')
    plt.axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.2f}%')
    plt.title("График точности на валидации (определение стиля)")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("style_accuracy.png")
    plt.close()
    print("График точности сохранён: style_accuracy.png")
    
    class_names = style_names_list
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Confusion Matrix (Стили) - Лучшая точность: {best_acc:.2f}%")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), 
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.xlabel("Предсказано")
    plt.ylabel("Истина")
    plt.tight_layout()
    plt.savefig("style_confusion_matrix.png")
    plt.close()
    print("Confusion matrix сохранена: style_confusion_matrix.png")
    
    with open("style_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(all_labels, all_predictions, target_names=class_names))
    print("Отчёт сохранён: style_report.txt")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")
    
    style_names = list(idx_to_style.values())
    
    def predict(image, direction=None):
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            probs = {style_names[i]: probabilities[0][i].item() for i in range(num_styles)}
        
        # Если передано направление, фильтруем стили
        if direction is not None:
            allowed_styles = DIRECTION_TO_STYLES.get(direction, [])
            if allowed_styles:
                probs = {s: p for s, p in probs.items() if s in allowed_styles}
                total = sum(probs.values())
                if total > 0:
                    probs = {s: p / total for s, p in probs.items()}
        
        return probs
    
    return predict

def findOS():
    base_path = os.path.dirname(__file__)
    DATASET_PATH = os.path.join(base_path, "НСвОИиТ курсовая датасет_converted")
    LABELS_FILE = os.path.join(DATASET_PATH, "labels_remapped.json")
    MODEL_PATH = os.path.join(base_path, "os_model_best.pth")
    
    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Словарь: стиль → список ОС
    STYLE_TO_OS = {
        "Apple Skeuomorphism": [
            "iPhone OS",
            "iOS 6",
            "Mac OS X 10.0 Cheetah",
            "Mac OS X 10.5 Leopard",
            "Mac OS X 10.9 Mavericks"
        ],
        "Windows Aero": [
            "Windows Vista",
            "Windows 7"
        ],
        "Apple Flat by Jony Ive": [
            "iOS 7",
            "iOS 11",
            "iOS 18",
            "Mac OS X 10.10 Yosemite",
            "macOS 10.15 Catalina",
            "macOS 11 Big Sur"
        ],
        "Windows Fluent": [
            "Windows 11"
        ],
        "Windows Metro": [
            "Windows 10"
        ],
        "Apple Liquid Glass": [
            "iOS 26",
            "macOS 26 Tahoe"
        ],
        "Windows 80-90s Retro": [
            "Windows 3.0",
            "Windows 98"
        ]
    }
    
    # Обратный словарь: ОС → стиль (для проверки)
    OS_TO_STYLE = {}
    for style, os_list in STYLE_TO_OS.items():
        for os_name in os_list:
            OS_TO_STYLE[os_name] = style
    
    print(f"\nУстройство: {DEVICE}")
    
    if os.path.exists(MODEL_PATH):
        print(f"\nНайдена сохранённая модель: {MODEL_PATH}")
        answer = input("Загрузить сохранённую модель? (y/n): ")
        if answer.lower() == 'y':
            train_loader, val_loader, num_os, idx_to_os = loadOS(
                labels_file=LABELS_FILE,
                root_dir=DATASET_PATH,
                batch_size=BATCH_SIZE,
                train_ratio=0.8,
                image_size=IMAGE_SIZE
            )
            
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_os)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = model.to(DEVICE)
            
            print(f"Модель загружена: {MODEL_PATH}")
            
            os_names = list(idx_to_os.values())
            
            def predict(image, style=None):
                model.eval()
                transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    probs = {os_names[i]: probabilities[0][i].item() for i in range(num_os)}
                
                # Если передан стиль, фильтруем ОС
                if style is not None:
                    allowed_os = STYLE_TO_OS.get(style, [])
                    if allowed_os:
                        probs = {os: p for os, p in probs.items() if os in allowed_os}
                        total = sum(probs.values())
                        if total > 0:
                            probs = {os: p / total for os, p in probs.items()}
                
                return probs
            return predict
    
    train_loader, val_loader, num_os, idx_to_os = loadOS(
        labels_file=LABELS_FILE,
        root_dir=DATASET_PATH,
        batch_size=BATCH_SIZE,
        train_ratio=0.8,
        image_size=IMAGE_SIZE
    )
    
    all_val_os = []
    for _, os_labels in val_loader:
        all_val_os.extend(os_labels.numpy())
    
    unique_os = sorted(set(all_val_os))
    os_names_list = [idx_to_os[i] for i in unique_os]
    
    plt.figure(figsize=(14, 6))
    plt.hist(all_val_os, bins=len(unique_os), alpha=0.7, color='blue', edgecolor='black')
    plt.xticks(unique_os, os_names_list, rotation=90, fontsize=8)
    plt.title("Распределение ОС в валидационной выборке")
    plt.xlabel("ОС")
    plt.ylabel("Количество иконок")
    plt.tight_layout()
    plt.savefig("os_distribution.png")
    plt.close()
    print("Гистограмма ОС сохранена: os_distribution.png")
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_os)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    print("\nОбучение модели для определения ОС...")
    print(f"Количество классов ОС: {num_os}")
    
    train_losses = []
    val_accuracies = []
    best_acc = 0
    best_epoch = 0
    best_model_state = None
    patience = 5
    patience_counter = 0
    all_predictions = []
    all_labels = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, os_labels in train_loader:
            images = images.to(DEVICE)
            os_labels = os_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, os_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        correct = 0
        total = 0
        epoch_predictions = []
        epoch_labels = []
        
        with torch.no_grad():
            for images, os_labels in val_loader:
                images = images.to(DEVICE)
                os_labels = os_labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += os_labels.size(0)
                correct += (predicted == os_labels).sum().item()
                
                epoch_predictions.extend(predicted.cpu().numpy())
                epoch_labels.extend(os_labels.cpu().numpy())
        
        acc = 100 * correct / total
        val_accuracies.append(acc)
        
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            all_predictions = epoch_predictions
            all_labels = epoch_labels
            patience_counter = 0
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% – НОВЫЙ ЛУЧШИЙ")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% (лучший: {best_acc:.2f}%)")
        
        if patience_counter >= patience:
            print(f"\nРанняя остановка на эпохе {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nЗагружена лучшая модель с эпохи {best_epoch}, точность {best_acc:.2f}%")
    
    print(f"\nФинальная точность: {best_acc:.2f}%")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', linewidth=2, markersize=4)
    plt.title("График потерь при обучении (определение ОС)")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("os_loss.png")
    plt.close()
    print("График потерь сохранён: os_loss.png")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', linewidth=2, markersize=4, color='green')
    plt.axhline(y=best_acc, color='r', linestyle='--', label=f'Best: {best_acc:.2f}%')
    plt.title("График точности на валидации (определение ОС)")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("os_accuracy.png")
    plt.close()
    print("График точности сохранён: os_accuracy.png")
    
    class_names = [idx_to_os[i] for i in range(num_os)]
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(min(20, num_os * 0.6), min(16, num_os * 0.5)))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Confusion Matrix (ОС) - Лучшая точность: {best_acc:.2f}%")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, ha='right', fontsize=7)
    plt.yticks(tick_marks, class_names, fontsize=7)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                plt.text(j, i, str(cm[i, j]), 
                         horizontalalignment="center",
                         fontsize=6,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.xlabel("Предсказано")
    plt.ylabel("Истина")
    plt.tight_layout()
    plt.savefig("os_confusion_matrix.png")
    plt.close()
    print("Confusion matrix сохранена: os_confusion_matrix.png")
    
    # Частые ошибки
    from collections import Counter
    print("10 самых частых ошибок")
    
    errors = []
    for true, pred in zip(all_labels, all_predictions):
        if true != pred:
            errors.append((true, pred))
    
    error_pairs = Counter(errors)
    for (true, pred), count in error_pairs.most_common(10):
        print(f"  {class_names[true]} → {class_names[pred]}: {count} раз(а)")
    
    with open("os_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(all_labels, all_predictions, target_names=class_names))
    print("\nПолный отчёт сохранён: os_report.txt")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")
    
    os_names = list(idx_to_os.values())
    
    def predict(image, style=None):
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            probs = {os_names[i]: probabilities[0][i].item() for i in range(num_os)}
        
        # Если передан стиль, фильтруем ОС
        if style is not None:
            allowed_os = STYLE_TO_OS.get(style, [])
            if allowed_os:
                probs = {os: p for os, p in probs.items() if os in allowed_os}
                total = sum(probs.values())
                if total > 0:
                    probs = {os: p / total for os, p in probs.items()}
        
        return probs
    
    return predict

def AI():
    base_path = os.path.dirname(__file__)
    DATASET_PATH = os.path.join(base_path, "НСвОИиТ курсовая датасет_converted")
    LABELS_FILE = os.path.join(DATASET_PATH, "labels_remapped.json")
    
    predict_direction = findDirection()
    predict_style = findStyle()
    predict_os = findOS()
    
    images, paths = user()
    
    if images:
        print(f"\nКлассификация пользовательских иконок:\n")
        
        for img, path in zip(images, paths):
            filename = os.path.basename(path)
            
            # Шаг 1: Определяем направление
            direction_probs = predict_direction(img)
            best_direction = max(direction_probs, key=direction_probs.get)
            
            # Шаг 2: Определяем стиль с учётом направления
            style_probs = predict_style(img, direction=best_direction)
            best_style = max(style_probs, key=style_probs.get) if style_probs else None
            
            # Шаг 3: Определяем ОС с учётом стиля
            os_probs = predict_os(img, style=best_style)
            best_os = max(os_probs, key=os_probs.get) if os_probs else None
            
            print(f"\n{filename}")
            print("-" * 50)
            print(f"  Направление дизайна: {best_direction} (уверенность {direction_probs[best_direction]*100:.1f}%)")
            
            if best_style:
                print(f"  Стиль: {best_style} (уверенность {style_probs[best_style]*100:.1f}%)")
            
            if best_os:
                print(f"  ОС: {best_os} (уверенность {os_probs[best_os]*100:.1f}%)")
            
            # Детализация
            sorted_directions = sorted(direction_probs.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  Детализация по направлениям:")
            for i, (direction, prob) in enumerate(sorted_directions):
                print(f"    {i+1}. {direction}: {prob*100:.1f}%")
            
            if style_probs:
                sorted_styles = sorted(style_probs.items(), key=lambda x: x[1], reverse=True)
                print(f"\n  Детализация по стилям (топ-5):")
                for i, (style, prob) in enumerate(sorted_styles[:5]):
                    print(f"    {i+1}. {style}: {prob*100:.1f}%")
            
            if os_probs:
                sorted_os = sorted(os_probs.items(), key=lambda x: x[1], reverse=True)
                print(f"\n  Детализация по ОС (топ-5):")
                for i, (os_name, prob) in enumerate(sorted_os[:5]):
                    print(f"    {i+1}. {os_name}: {prob*100:.1f}%")
            
    else:
        print("\nНет изображений для классификации")
        print("Положите иконки в папку User на рабочем столе и запустите снова")

AI()