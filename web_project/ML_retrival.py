import pandas as pd
import numpy as np
import string
import os
from rank_bm25 import BM25Okapi
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import queries


folder_path = 'archive'
df = pd.DataFrame()

for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        full_path = os.path.join(folder_path, file)
        data = pd.read_csv(full_path)
        df = pd.concat([df, data], ignore_index=True)

df['discount_price'] = (
    df['discount_price']
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .replace(['', 'NaN', 'null', 'None'], np.nan)
    .astype(float)
    .fillna(0)
)

df['actual_price'] = (
    df['actual_price']
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .replace(['', 'NaN', 'null', 'None'], np.nan)
    .astype(float)
    .fillna(0)
)

df['ratings'] = (
    df['ratings']
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .replace(['', 'NaN', 'null', 'None', 'Get', 'FREE'], np.nan)
    .astype(float)
    .fillna(0)
)


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens


df['tokenized_name'] = df['name'].apply(preprocess_text)
df['tokenized_main_category'] = df['main_category'].apply(preprocess_text)
df['tokenized_sub_category'] = df['sub_category'].apply(preprocess_text)

tokenized_products = [t_p for t_p in df['tokenized_name']]
tokenized_main_category = [t_p for t_p in df['tokenized_main_category']]
tokenized_sub_category = [t_p for t_p in df['tokenized_sub_category']]


all_search_queries = [
    # Electronics (25)
    "wireless headphones", "bluetooth earbuds", "smartphone android", "apple iphone",
    "smart watch", "fitness tracker", "bluetooth speaker", "portable power bank",
    "usb-c charger", "tablet case", "gaming mouse", "mechanical keyboard", "4k monitor",
    "laptop for work", "drawing tablet", "e-book reader", "smart watch with ECG",
    "wireless charger", "bluetooth headset", "streaming webcam", "recording microphone",
    "portable SSD", "wi-fi router", "smart speaker", "VR headset",

    # Home & Kitchen (20)
    "coffee maker", "air fryer", "instant pot", "kitchen blender", "stand mixer",
    "food processor", "toaster oven", "microwave oven", "vacuum cleaner", "air purifier",
    "humidifier", "essential oil diffuser", "bed sheet set", "memory foam pillow",
    "weighted blanket", "cookware set", "kitchen knives", "non-stick pan", "baking sheets",
    "food storage containers",

    # Fashion & Clothing (15)
    "running shoes", "sneakers for men", "women's dress", "jeans for women", "winter jacket",
    "rain coat", "athletic shorts", "yoga pants", "business shirt", "casual t-shirt",
    "wool socks", "swimwear", "handbag for women", "backpack for laptop", "sunglasses polarized",

    # Books & Media (15)
    "science fiction books", "romance novels", "business books", "cookbook for beginners",
    "children's story books", "self-help books", "programming textbooks", "fantasy book series",
    "mystery thriller", "audio books subscription", "kindle unlimited", "coloring books for adults",
    "educational books for kids", "biography of entrepreneurs", "graphic novels",

    # Toys & Games (10)
    "lego sets for kids", "board games for family", "educational toys", "video games ps5",
    "nintendo switch games", "outdoor play equipment", "puzzle games", "action figures",
    "stem learning toys", "baby toys",

    # Sports & Outdoors (10)
    "yoga mat", "dumbbell set", "exercise bike", "camping tent", "hiking backpack",
    "fishing rod", "bicycle for adults", "tennis racket", "basketball shoes", "water bottle insulated",

    # Beauty & Personal Care (5)
    "skincare set", "hair dryer", "electric toothbrush", "perfume for women", "makeup brush set"
]

# df = pd.read_csv("data.csv")

model = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)


def normalize_scores(scores):
    scores_array = np.array(scores)
    if len(scores_array) == 0:
        return scores_array

    min_score = np.min(scores_array)
    max_score = np.max(scores_array)

    if max_score == min_score:
        return np.full_like(scores_array, 0.5)
    return (scores_array - min_score) / (max_score - min_score)


def assign_relevance_batch(scores_name, scores_main_cat, scores_sub_cat, ratings):
    text_scores = 0.5 * scores_name + 0.2 * scores_main_cat + 0.2 * scores_sub_cat
    quality_scores = 0.1 * (ratings / 100.0)
    total_scores = text_scores + quality_scores

    conditions = [
        total_scores >= 0.65,
        (total_scores >= 0.3) & (total_scores < 0.65),
        total_scores < 0.3
    ]
    choices = [2, 1, 0]  # 2=Полностью релевантен, 1=Частично, 0=Не релевантен

    return np.select(conditions, choices)


def prepare_data(df, queries, n_best=1000):
    all_features = []
    all_labels = []
    tokenized_names = [t_p for t_p in df['tokenized_name']]
    tokenized_main_cat = [t_p for t_p in df['tokenized_main_category']]
    tokenized_sub_cat = [t_p for t_p in df['tokenized_sub_category']]

    bm25_name = BM25Okapi(tokenized_names)
    bm25_main_category = BM25Okapi(tokenized_main_cat)
    bm25_sub_category = BM25Okapi(tokenized_sub_cat)

    for i, query in enumerate(queries):
        print(f"Обработка запроса {i+1}/{len(queries)}")
        tokenized_query = preprocess_text(query)

        scores_name = bm25_name.get_scores(tokenized_query)
        scores_main_cat = bm25_main_category.get_scores(tokenized_query)
        scores_sub_cat = bm25_sub_category.get_scores(tokenized_query)
        score_name_norm = normalize_scores(scores_name)
        score_main_cat_norm = normalize_scores(scores_main_cat)
        score_sub_cat_norm = normalize_scores(scores_sub_cat)
        ratings = df['ratings'].values

        features = np.column_stack([score_name_norm, score_main_cat_norm, score_sub_cat_norm, ratings])

        labels = assign_relevance_batch(score_name_norm, score_main_cat_norm, score_sub_cat_norm, ratings)

        n_samples = min(n_best, len(features))
        random_indices = np.random.choice(len(features), size=n_samples, replace=False)

        features_random = features[random_indices]
        labels_random = labels[random_indices]

        all_features.append(features_random)
        all_labels.append(labels_random)

    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    return X, y


def create_dataloaders(X, y, batch_size=256, train_ratio=0.9, val_ratio=0.15):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, mode="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if mode == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            if mode == "cuda":
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                if mode == "cuda":
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.2f}%')
            print('---')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }


X, y = prepare_data(df, all_search_queries, n_best=10000)
train_loader, val_loader, test_loader = create_dataloaders(X, y)
history = train_model(model, train_loader, val_loader, num_epochs=10, mode="cuda")
print(history)
torch.save(model.state_dict(), "model_state_dict.pth")
# df.to_csv("data.csv", index=False)
