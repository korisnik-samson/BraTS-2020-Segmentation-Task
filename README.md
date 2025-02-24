# NOTEBOOK INSIGHTS:

## Data Processing:
* CSV file creation (links.csv) that maps file paths for the BraTS dataset.
    - `root_list`: names of the directories containing the individual patient data.<br>
      `tot_list`: lists of file paths for each patient (flair, seg, t1, t1ce, t2).

    - Patient Directory Iterations:
        - `os.listdir(root_df)`: to provide a list of patient directories.
        - `os.path.join(root_df, filename_root)`: to create a sub-path for each patient directory.
        - `np.sort(...)`: to sort the list of patient directories alphabetically.
        - `tdqm(...)`: to provide a progress bar for the iteration.
  
    - File Iterations for each Patient:
      - `subpath = os.path.join(root_df, filename_root)`: to build the full path to the directory for a specific patient.
      - After which, the file paths for each patient are appended to the `file_list` list.
      - A for loop is used to iterate over the files in the patient directory.
      - `os.path.join(subpath, filename)`: to create the full path to the file_list.
  
    - Dataframe Creation:
      - `pd.DataFrame(root_list, columns=['DIR'])`: to create a dataframe with the patient directory names.
      - `pd.DataFrame(tot_list, columns=['flair', 'seg', 't1', 't1ce', 't2'])`: to create a dataframe with the file paths for each patient.
      - `pd.concat(...)`: to concatenate the two dataframes into a single dataframe.
      - `axis=1`: to concatenate the dataframes along the columns.
        
    ```python
    root_list = []
    tot_list = []
    
    for filename_root in tqdm(np.sort(os.listdir(root_df))[:-2]):
        subpath = os.path.join(root_df, filename_root)
        file_list = []
    
        for filename in np.sort(os.listdir(subpath)):
            file_list.append(os.path.join(subpath, filename))
    
        root_list.append(filename_root)
        tot_list.append(file_list)
    
    maps = pd.concat([
    pd.DataFrame(root_list, columns=['DIR']),
    pd.DataFrame(tot_list, columns=['flair', 'seg', 't1', 't1ce', 't2'])
    
    ], axis=1)
    
    maps.to_csv('scratch/links.csv', index=False)
    ```
* Dictionary Population with file paths for different MRI modalities (seg, t1, t1ce, t2, flair) for each patient in the training dataset.
    - `image_path_dict`: dictionary to store the file paths for each patient and their MRI modalities `seg, flair, t1, t1ce, t2`.
    - 