import ipdb
from cleanlab_studio import Studio
import numpy as np
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    api_key = 'd2b832d47ac74ec0b2db401620875872'
    cleanset_id = 'c7edd976e6f64eb7823e1252fb8a3921' # Fast cleaning
    #cleanset_id = 'b4b30a0e602b465a99e8c8601dbe305e' # Deep cleaning
    studio = Studio(api_key)

    cleanset = studio.download_cleanlab_columns(cleanset_id)
    print(cleanset.head())

    # Calculate noisy label detection accuracy 
    ###########################################
    corruption_rate = 0.40
    dataset_size =  len(cleanset)
    nr_corrupt_instances = int(dataset_size * corruption_rate)

    corrupt_gt, corrupt_prediction = [], []

    # Note: All samples with an index < nr_corrupt_instances were corrupt
    for idx in range(dataset_size):
        sample_dataset = cleanset.iloc[idx]
        file_name, label_issue, ambigious = sample_dataset['id'], sample_dataset['is_label_issue'], sample_dataset['is_ambiguous']
        outlier = sample_dataset['is_outlier'] 

        _corrupt_predicton = True if (label_issue or ambigious or outlier) else False

        # Compute ground-truth from filename
        idx_orig_dataset = int(file_name.split('/')[-1].split('.png')[0]) # '0/10037.png' --> 10037
        _corrupt_gt = True if idx_orig_dataset < nr_corrupt_instances else False

        corrupt_gt.append(_corrupt_gt)
        corrupt_prediction.append(_corrupt_predicton)

    cm = confusion_matrix(corrupt_gt, corrupt_prediction)

    #Now the normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm.diagonal()

    print(f'Accuracy to detect clean-samples: {100*cm.diagonal()[0]:0.2f}%')
    print(f'Accuracy to detect noisy-samples: {100*cm.diagonal()[1]:0.2f}%')

    corrupt_gt = np.array(corrupt_gt)
    corrupt_prediction = np.array(corrupt_prediction)
    overall_acc = 100*np.mean(corrupt_gt == corrupt_prediction)
    print(f'Overall accuracy: {overall_acc:0.2f}%')

    # Accuracy to detect noisy samples in the corrupt set
    #accuracy_noisy_detect = (np.array(clean_lab_result['corrupt']) == True).mean() # 90.3% 
    #accuracy_clean_detect = (np.array(clean_lab_result['clean']) == False).mean()  # 88.6% 
        