import numpy as np
import pandas as pd
import config

class KNN:
    def __init__(self, k, dataset, X, y):
        self.k = k
        self.dataset = dataset
        self.X = X
        self.y = y
    
    def euclidean_distance(self, a1, a2):
        return np.sqrt(np.sum(np.power((a1 - a2), 2)))
    
    def most_common(self, labels):
        label_counts = {}
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        max_count = -1
        most_common_label = None
        for label, count in label_counts.items():
            if count > max_count:
                max_count = count
                most_common_label = label

        return most_common_label

    def predict(self, target, X_train, y_train):
        near_x_distances = [self.euclidean_distance(target, x) for x in X_train]
        k_near_indices = np.argsort(near_x_distances)[:self.k]
        k_near_labels = [y_train[i] for i in k_near_indices]        
        return self.most_common(k_near_labels)
    
    def vectorized_predict(self, target, X_train, y_train):
        near_x_distances = np.linalg.norm(X_train - target, axis=1)
        k_near_indices = np.argsort(near_x_distances)[:self.k]
        k_near_labels = [y_train[i] for i in k_near_indices]        
        return self.most_common(k_near_labels)

    def leave_one_out(self):
        predictions = []
        for i in range(len(self.X)):
            target = self.X.iloc[i].values
            X_train = self.X.drop(index=i).values
            y_train = self.y.drop(index=i).values
            predictions.append(self.predict(target, X_train, y_train))
        return predictions

    def caculte_error(self, predictions, classes, classifier_name):
        if 'Normal' in config.train_file:
            dataset_name = config.test_file.replace('.txt', '')
        else:
            dataset_name = config.train_file.replace('.txt', '')

        print(f'\nEvaluation of {classifier_name} for calssification of the {dataset_name} dataset')
        
        index = 0
        num_of_errors = 0
        for c_class in classes:
            num_of_c_class = len(self.dataset[self.dataset['class']==c_class])
            error = 0
            for i in range(num_of_c_class):
                if int(predictions[index+i]) != int(c_class):
                    error += 1
            index += num_of_c_class
            num_of_errors += error
            print(f'Number Of Errors Occured In Class {c_class}: {error}')
        
        print(f'\nError Rate: {num_of_errors/len(predictions):0.3f}')

    def test_knn_classifier(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            target = X_test.iloc[i].values
            X_train = self.X.values
            y_train = self.y.values

            predictions.append(self.vectorized_predict(target, X_train, y_train))
        return predictions

class MMDC:
    def __init__(self, classes, dataset, X, y):
        self.class_mean = {}
        for clas in classes:
            self.class_mean[clas] = np.mean(X[y == clas])
        self.X = X
        self.y = y
        self.dataset = dataset
    
    def euclidean_distance(self, a1, a2):
        return np.sqrt(np.sum(np.power((a1 - a2), 2)))
    
    def predict(self, target):
        min_distance = np.inf
        assigned_label = None
        for clas in self.class_mean.keys():
            distance_to_mean = self.euclidean_distance(target, self.class_mean[clas])
            if distance_to_mean < min_distance:
                min_distance = distance_to_mean
                assigned_label = clas
        return assigned_label

    def leave_one_out(self):
        predictions = []
        for i in range(len(self.X)):
            target = self.X.iloc[i].values
            X_train = self.X.drop(index=i).values
            y_train = self.y.drop(index=i).values
            predictions.append(self.predict(target))
        return predictions
    
    def caculte_error(self, predictions, classifier_name):
        if 'Normal' in config.train_file:
            dataset_name = config.test_file.replace('.txt', '')
        else:
            dataset_name = config.train_file.replace('.txt', '')

        print(f'\nEvaluation of {classifier_name} for calssification of the {dataset_name} dataset')
        
        index = 0
        num_of_errors = 0
        for c_class in classes:
            num_of_c_class = len(self.dataset[self.dataset['class']==c_class])
            error = 0
            for i in range(num_of_c_class):
                if int(predictions[index+i]) != int(c_class):
                    error += 1
            index += num_of_c_class
            num_of_errors += error
            print(f'Number Of Errors Occured In Class {c_class}: {error}')
        
        print(f'\nError Rate: {num_of_errors/len(predictions):0.3f}')
    
    def test_mmdc_classifier(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            target = X_test.iloc[i].values
            predictions.append(self.predict(target))
        return predictions

if __name__ == "__main__":
## ******************************** For IRIS and Liquid Dataset ******************************************
    # data_pd = pd.read_csv(config.train_file, delim_whitespace=True, header=None, names=config.data_columns)
    # df = pd.DataFrame(data_pd)
    # classes = df['class'].unique()
    # y = df['class']
    # X = df.drop(columns=['class'])

    #                     # ******************* k Nearest Neighbors Classifier ******************************
    # knn = KNN(config.k, df, X, y)
    # preds = knn.leave_one_out()
    # knn.caculte_error(preds, classes, 'k Nearest Neighbors Classifier')
                        
    #                     # ******************* Minimum Mean Distance Classifier ******************************
    # mmdc = MMDC(classes, df, X, y)
    # preds = mmdc.leave_one_out()
    # mmdc.caculte_error(preds, 'Minimum Mean Distance Classifier')



## ***************************************For Normal Dataset ************************************************** #

    train_file = 'Normal-Data-Training_dat.txt'
    test_file = 'Normal-Data-Testing_dat.txt'
    data_pd = pd.read_csv(config.train_file, delim_whitespace=True, header=None, names=config.data_columns)
    train_df = pd.DataFrame(data_pd)

    y_train = train_df['class']
    X_train = train_df.drop(columns=['class'])
    classes = train_df['class'].unique()

    data_pd = pd.read_csv(config.test_file, delim_whitespace=True, header=None, names=config.data_columns)
    test_df = pd.DataFrame(data_pd)

    y_test = test_df['class']
    X_test = test_df.drop(columns=['class'])

                        ## ******************* k Nearest Neighbors Classifier ******************************
    knn = KNN(config.k, train_df, X_train, y_train)
    predictions = knn.test_knn_classifier(X_test)
    knn.caculte_error(predictions, classes, "'k Nearest Neighbors Classifier'")
                        
                        ## ******************* Minimum Mean Distance Classifier ******************************
    mmdc = MMDC(classes, train_df, X_train, y_train)
    predictions = mmdc.test_mmdc_classifier(X_test)
    mmdc.caculte_error(predictions, "'Minimum Mean Distance Classifier'")


