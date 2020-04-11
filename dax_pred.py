class Predictor (object):
     timestamp = 60
     dataset = np.array
     sc = MinMaxScaler(feature_range = (0, 1))
+    sc_out = MinMaxScaler(feature_range = (0, 1))
     dimension = 0
     model = []
 
     def __init__(self,offset_file, predict_file, x_columns_names, y_column_name):
        offset = self.getDataset(offset_file, x_columns_names)
        prediction_set = self.getDataset(predict_file, x_columns_names)
        dataset = offset.append(prediction_set)
        inputs = dataset[len(dataset) - len(prediction_set) - self.timestamp:].values
-       self.sc.fit(inputs[:,0].reshape(-1,1))       
-       #self.sc.fit(self.dataset.values.reshape(-1,1))
+       self.sc_out.fit(inputs[:,0].reshape(-1,1))
+       #self.sc.fit(inputs[:,0].reshape(-1,1))
+       self.sc.fit(dataset.values.reshape(-1,1))
        self.dimension = len(x_columns_names)       
        self.model = self.createModel(self.initX(inputs))
 
     def predict(self, regressor):
         regressor = self.loadPredictor(regressor)

     def initX(self,inputs):
        for i in range (0, self.dimension):
            inputs = np.append(inputs, self.scale(inputs[:,i]), axis = 1)
        for i in range (0, self.dimension):    
            inputs = np.delete(inputs,0,axis = 1)
+       """    
+       data = np.asarray(inputs)
+       data = self.sc.fit_transform(data)
+       """
        return inputs
    
     def getDataset(self,file, x_columns_names):
         dataset = pd.read_csv(file, sep= ',')
         # dataset = pd.read_csv(file, sep= ',', index_column = 0)

     def scale(self, unscaled):
         scaled = self.sc.transform(unscaled.reshape(-1,1))
         return scaled
     
     def unscale(self, scaled):
        unscaled = self.sc_out.inverse_transform(scaled)
        return unscaled
     
     def loadPredictor(self,name):
         reg = 'regressors/'+name+ '.json'
         weights = 'regressors/'+name+ '.h5'
         plt.grid(b=None, which='major', axis='both')
         plt.title('DAX Stock Close Price Prediction')
         plt.legend()
         plt.show()
 
-offset_file = './daten/offset_2019.csv'
-predict_file = './daten/predict_05.csv'
-input_columns = ['Open','Close']
+offset_file = './daten/offset_2013.csv'
+predict_file = './daten/predict.csv'
+#input_columns = ['Open','Close']
+input_columns = ['Close']
 output_column = 'Close'
 pr = Predictor(offset_file,predict_file,input_columns,output_column)
-#name = 'dax_regressor'
+name = 'dax_regressor'
 #name = 'dax_2_regressor'
-name = 'd_2_l_4_u_60_dax_regressor'
+#name = 'd_1_l_3_u_75_a_adadelta_er_mean_squared_logarithmic_error_ep_150_b_32_dax_regressor'
+#name = 'd_1_l_4_u_75_a_rmsprop_er_mean_squared_logarithmic_error_ep_150_b_32_dax_regressor'
+#name = 'd_1_l_4_u_75_a_adam_er_mean_squared_logarithmic_error_ep_150_b_32_dax_regressor'
 output = pr.predict(name)
 
 inputs = pr.getDataset(predict_file,input_columns)
 pr.plot(output, inputs)
 inputs = pd.DataFrame(inputs)
 output = pd.DataFrame(output)
 
-"""
-with pd.ExcelWriter('./prediction/result.xlsx', engine='openpyxl', mode='a') as writer:
-    #inputs.to_excel(writer, sheet_name='input')
-    output.to_excel(writer, sheet_name= name)
-"""
\ No newline at end of file
+with pd.ExcelWriter('./prediction/'+name+'.xlsx', engine='openpyxl', mode='a') as writer:
+    inputs.to_excel(writer, sheet_name='input')
+    output.to_excel(writer, sheet_name= 'prediction')
\ No newline at end of file
