
        numChars = len(characteristics[options[j]]) #save num of how many characteristics are being used for building model
        templistofChars = characteristics[options[j]]
        templistofChars.append('Close')


        #print("This is the temp list of attributes: ",templistofChars)

        TempDataFrame = pd.DataFrame(dataset[templistofChars])
        #print(TempDataFrame)

        TempDataFrame = TempDataFrame.dropna() # remove all NaNs, due to these being incomplete data sets

        output_var = pd.DataFrame(TempDataFrame['Close'])  ## Select the close value as the output data set, this will be used for "Y"

        #print("TempDataFrame:")
        #print(TempDataFrame['2022':'2022'])
        #print("\n","This is the output_var, aka the y values")
        #print(output_var)
        TempDataFrame = TempDataFrame.drop("Close", axis=1)
        Y_train = output_var[:'2021'].values ## grab training data from outputvar
        Y_test = output_var['2022':'2022'].values #grab test data from outputvar

        #print(i)
        #print(options[j])
        #print(X_Values)

        X_Values = TempDataFrame #pd.DataFrame(TempDataFrame[characteristics[options[j]]])
        #*****************
        #change code here

        #print("This is the dataframe of Xvalues")
        #print(X_Values)

        X_train = X_Values[:'2021'].values
        X_test = X_Values['2022':'2022'].values  ## grouping day of year in and scaling...

        #print(X_test)
        '''
        Y_train = output_var[:'2021'].values ## grab training data from outputvar
        Y_test = output_var['2022':'2022'].values #grab test data from outputvar
    
        #print(i)
        #print(options[j])
        #print(X_Values)
    
        X_Values = pd.DataFrame( dataset[characteristics[options[j]]])
        print(X_Values)
        X_train = X_Values[:'2021'].values
        X_test = X_Values['2022':'2022'].values  ## grouping day of year in and scaling...
        #print(X_test)
    
        #*************************************************************************************************************
        #This section will take the Test set and build an array of each data point that is looking back the
        # number of specified days, it will take in the number of chars being used so it can shape properly
        # the structure will be as  (size , #lookbackdays, chars )
        # each lookback set has a corresponding Y value, this is in order to handle the offest of looking back
        X_test_lookback = []
        Y_test_ready = []
        '''

        X_test_lookback, Y_test_ready = LookbackTest(X_test, Y_test, X_train, numChars, lookback_days)


        #print("Temp data frame",TempDataFrame['2022':'2022'])
        #print("X test lookback", X_test_lookback)
        #print("Y test ready", Y_test_ready)


        #*************************************************************************************************************
        #This section will take the Training set and build an array of each data point that is looking back the
        # number of specified days, it will take in the number of chars being used so it can shape properly
        # the structure will be as  (size , #lookbackdays, chars )
        # each lookback set has a corresponding Y value, this is in order to handle the offest of looking back

        #X_train_lookback, Y_train_ready = LookbackTrain(X_train, Y_train, numChars , lookback_days)

        X_train_lookback = []
        Y_train_ready = []
        #print(training_set_scaled[54-54:54, 1])

        for i in range(lookback_days,len(X_train)): ## *************double check this len statement
            X_train_lookback.append(X_train[i-lookback_days:i])
            Y_train_ready.append(Y_train[i,0])

        X_train_lookback, Y_train_ready = np.array(X_train_lookback), np.array(Y_train_ready)
        #print(X_train_lookback)
        # Reshaping X_train for efficient modeling
        X_train_lookback = np.reshape(X_train_lookback, (X_train_lookback.shape[0],X_train_lookback.shape[1],numChars))

        #print("Xtrain lookback", X_train_lookback)
        #print("YTrain ready", Y_train_ready)
    #*************************************************************************************************************


        #print(X_train_lookback)
        #print(y_train)
        #print(numChars)   #verification step to prove the array dimensions
        #print("xtrain shape", X_train_lookback.shape) #verification step to prove the array dimensions
        #print("Y_train_ready shape", Y_train_ready.shape)

        #**************************************************************************************************************
        #This section handles building the model and then testing it out.


        #break

        Model1 = GRUModelBuild(X_train_lookback, Y_train_ready, numChars, lookback_days) #call the builder function above
        #Model1.summary()  ## This will print the structure of the GRU that has been built


        Model1.fit(X_train_lookback, Y_train_ready, validation_split=0.1, epochs = 5) # ,batch_size=1


        #train the model using the datasets, validation split will take a random 10% of the data out of training and use it
            # to test the model for validation, the result will show if genrealization is occuring properly
            # train over 15 iteraitions (epochs)

        GRU_predicted_stock_price = Model1.predict(X_test_lookback)
        #Predict the prices using the test set with lookback applied to it

        # Visualizing the results for GRU
        GRU_predicted_stock_price = Descaler(GRU_predicted_stock_price, Factor)
        Y_test_ready = Descaler(Y_test_ready, Factor)
        ## Descale the predictions, Y_test_ready was used for validation of descaling.

        Y_test = Descaler(Y_test, Factor)

        plot_predictions(Y_test, GRU_predicted_stock_price, lookback_days)
        #Plot the predictions, the final attribute of this function allows for shifting the plots to see what occured before
            # the displayed prediction


        # Evaluating GRU
        tempHold_RMSE = return_rmse(Y_test_ready, GRU_predicted_stock_price[:len(GRU_predicted_stock_price)-1]) #call rmse to give error on data set
        tempHold_DirErr = Directional_error(Y_test_ready, GRU_predicted_stock_price[:len(GRU_predicted_stock_price)-1])

        if(useMetric == "RMSE"):
            if((tempHold_RMSE < curr_BestResultRMSE) or ((tempHold_DirErr > curr_BestResultDirErr) and (tempHold_RMSE == curr_BestResultRMSE) ) ):
                curr_BestResultRMSE = tempHold_RMSE
                curr_BestResultDirErr = tempHold_DirErr
                attrib_used_metric = characteristics[options[j]]
                prGreen("This Model has been saved")
                Model1.save(model_SaveLocation)
                bestModelAttributes = characteristics[options[j]]
            else:
                print("This model was discarded")

        if(useMetric == "DirErr"):
            if((tempHold_DirErr > curr_BestResultDirErr) or ((tempHold_DirErr == curr_BestResultDirErr) and (tempHold_RMSE < curr_BestResultRMSE) ) ):
                curr_BestResultDirErr = tempHold_DirErr
                curr_BestResultRMSE = tempHold_RMSE
                attrib_used_metric = characteristics[options[j]]
                prGreen("This Model has been saved")
                Model1.save(model_SaveLocation)
                bestModelAttributes = characteristics[options[j]]
            else:
                print("This model was discarded")


        #CSV_Header = ['Ticker', 'Attributes', 'DirErr', 'RMSE']
        training_Data_forCSV.append({'Ticker': ticker_File.strip('.csv'), 'Attributes': characteristics[options[j]], 'DirErr': tempHold_DirErr, 'RMSE': tempHold_RMSE})
        print("Attributes:", characteristics[options[j]]) #print what characteristics were used

        #print(Model1.summary()) #this will print out the structure of the GRU

        keras.backend.clear_session()#delete the GRU and clear all weight data, this is essential to train a new model after
        print("Model Cleared")
