import pandas as pd
import numpy as np
import os

N = 16
data_folder = './splits/TCGA_LIVER/task_tcga_lIVER_subtyping_100/'
all_data = np.array(pd.read_csv('./create_csv/uuid_name_file.csv', skiprows=1, header=None))
save_folder = './fewshot_splits/TCGA_LIVER/'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


slidename2label = {}
for each_data in all_data:
    key = str(each_data[1]).strip().lower().rstrip('.svs')
    slidename2label[key] = each_data[-1]

print(f"Mapping size: {len(slidename2label)}")

for j in range(5):
    print(f"\n=== Split {j} ===")

    orginal_data_split_path = data_folder + f'/splits_{j}.csv'
    save_path = save_folder + f'/splits_{j}.csv'

    df = pd.read_csv(orginal_data_split_path)


    train_col = df['train']
    val_col = df['val']
    test_col = df['test']


    all_slide_label = []
    valid_indices = []

    for idx, slide in enumerate(train_col):
        if pd.isna(slide):
            continue

        slide_key = str(slide).strip().lower().rstrip('.svs')
        slide_label = slidename2label.get(slide_key, np.nan)

        if slide_key not in slidename2label:
            print(f"  Warning - slide '{slide_key}' not in mapping")

        all_slide_label.append(str(slide_label))
        valid_indices.append(idx)

    all_slide_label_arr = np.array(all_slide_label)
    unique_label = np.unique(all_slide_label_arr)

    print(f"Unique labels: {unique_label}")


    selected_train_slide = []

    for each_label in unique_label:
        if each_label == 'nan':
            continue

        each_index = np.where(all_slide_label_arr == each_label)[0]
        print(f"Label {each_label}: {len(each_index)} samples")

        if len(each_index) == 0:
            continue

        n_select = min(N, len(each_index))
        selected_index = np.random.choice(each_index, size=n_select, replace=False)

        for idx in selected_index:
            selected_train_slide.append(train_col.iloc[valid_indices[idx]])

    print(f"Selected train samples: {len(selected_train_slide)}")

    new_train_col = pd.Series(selected_train_slide)


    max_len = max(len(new_train_col), len(val_col), len(test_col))

    new_train_col = new_train_col.reindex(range(max_len))
    val_col = val_col.reindex(range(max_len))
    test_col = test_col.reindex(range(max_len))


    new_df = pd.DataFrame({
        'train': new_train_col,
        'val': val_col,
        'test': test_col
    })


    new_df.to_csv(save_path, index=False)

print("\nDone!")