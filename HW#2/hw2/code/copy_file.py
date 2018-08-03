for r in raw:
        point = {}
        point["label"] = (int(r['income'] == '>50K'))
        data.append(point)

        features = []
        features.append(1.)
        for i in range(1,len(feature_list)):
            features.append(r[feature_list[i]])
        processed_data.append(features)    

    df = pd.DataFrame(processed_data)
    df.columns = feature_list
    df_encode = pd.get_dummies(df, columns=discrete_list, drop_first=True)
    
    mean_train = np.asarray(df_encode, dtype=np.float).mean(axis=0)
    std_train = np.asarray(df_encode, dtype=np.float).std(axis=0)
    std_train[std_train == 0] = 1

    data_train = np.asarray(df_encode, dtype=np.float64)
    norm_train = (data_train - mean_train)/(std_train)  # normalize train/validation set by substracting train mean and dividing by train std
    for j in range(len(data)):
        transit = data[j]
        transit['features']=norm_train[j]
    return data


#Update model using learning rate and L2 regularization
#L2 norm is applied to trade off extreme large weights
#model is updated based on stochastic gradient ascent algorithm
def update(model, point, delta, rate, lam):
    update_delta = np.asarray(point)*delta
    regularization = -lam*np.asarray(model)
    gradient = regularization + update_delta
    model1 = model + rate*gradient
    return model1

#Train model using training data
#rate indicates learning rate, lam indicates penalty coefficient, epochs indicates iteration times
def train(data, epochs, rate, lam):
    model = initialize_model(len(data[0]['features']))
    for j in range(epochs):
        for i in range(len(data)):
            x = data[i]
            point = x['features']
            label = x['label']
            delta = label - logistic(dot(model,point))
            model = update(model, point, delta, rate, lam)
    return model
