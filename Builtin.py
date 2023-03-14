from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import LinearDiscriminantAnalysis


def get_LDA_Variation_accuracy():
    start_time = time.time()
    fld = LinearDiscriminantAnalysis(solver='svd')
    fld.fit(d_samples, y_samples)
    y_pred = fld.predict(d_test)
    accuracy = accuracy_score(y_test, y_pred)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"LDA Variation time: {elapsed_time:.2f} seconds")
    print(f"LDA accuracy: {accuracy}")
    
def get_PCA_Variation_accuracy():
    start_time = time.time()
    ipca = IncrementalPCA(n_components=4)
    for batch in np.array_split(d_samples,50):
            ipca.partial_fit(batch)
    trans_data = ipca.transform(d_samples)
    d_test_trans = ipca.transform(d_test)
    y_pre = get_predict(trans_data, d_test_trans, y_samples, 1)
    accuracy = accuracy_score(y_test, y_pre)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"PCA Variation time: {elapsed_time:.2f} seconds\nWith accuracy ={accuracy}")
    return np.array(accuracy)