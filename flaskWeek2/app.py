import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,send_from_directory
import pickle
import time
import os.path
from os import path
import os
# from werkzeug.utils import secure_filename
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


app = Flask(__name__)


import pickle
filename = r'C:\Users\ASUS\PycharmProjects\flaskWeek2\models\real_model2.sav'
nft_tree = pickle.load(open(filename, 'rb'))

def input(a,z,e,r,t,y,u,i,o,p,q,s,d,f,g,h):
  input={'asset_contract.created_date':a, 'asset_contract.name':z,
       'asset_contract.schema_name':e, 'asset_contract.asset_contract_type':r,
       'collection.created_date':t, 'collection.slug':y,
       'collection.safelist_request_status':u, 'owner.address':i, 'is_presale':o,
       'asset_contract.total_supply':p, 'asset_contract.seller_fee_basis_points':q,
       'asset_contract.dev_seller_fee_basis_points':s,
       'asset_contract.opensea_seller_fee_basis_points':d, 'collection.featured':f,
       'collection.dev_seller_fee_basis_points':g,
       'collection.opensea_seller_fee_basis_points':h }
  return input


def predictNFT(a, z, e, r, t, y, u, i, o, p, q, s, d, f, g, h):
    X1_test = pd.read_csv(r'C:\Users\ASUS\PycharmProjects\flaskWeek2\downloads\X_test.csv')
    X1_train = pd.read_csv(r'C:\Users\ASUS\PycharmProjects\flaskWeek2\downloads\X_train.csv')
    X1_test = X1_test.drop(['Unnamed: 0'], axis=1)
    X1_train = X1_train.drop(['Unnamed: 0'], axis=1)

    # converting the numerical variable
    nft_impute = SimpleImputer(strategy='median')
    x1_train_int = nft_impute.fit_transform(X1_train.select_dtypes(['int64', 'float64']))
    nft1_int_conv = pd.DataFrame(x1_train_int, columns=X1_train.select_dtypes(['int64', 'float64']).columns)

    ordinal_obj = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=9999)
    transformed1_obj = ordinal_obj.fit_transform(X1_train.select_dtypes('object'))
    nft1_or_encode = pd.DataFrame(transformed1_obj, columns=X1_train.select_dtypes('object').columns)

    train1_conv_nft_df = pd.concat([nft1_int_conv, nft1_or_encode], axis=1)

    dic = input(a, z, e, r, t, y, u, i, o, p, q, s, d, f, g, h)
    X1_test = X1_test.append(dic, ignore_index=True)

    x1_test_int = nft_impute.transform(X1_test.select_dtypes(['int64', 'float64']))
    nft1_int_conv_test = pd.DataFrame(x1_test_int, columns=X1_test.select_dtypes(['int64', 'float64']).columns)

    test_transformed1_obj = ordinal_obj.transform(X1_test.select_dtypes('object'))
    test1_nft_or_encode = pd.DataFrame(test_transformed1_obj, columns=X1_test.select_dtypes('object').columns)

    test1_nft_df = pd.concat([nft1_int_conv_test, test1_nft_or_encode], axis=1)

    inputToPredict = test1_nft_df.loc[2170].tolist()

    output = nft_tree.predict([inputToPredict])
    return (output[0])



# # imporing model and scalar object
# MODEL= 'my_model.pkl'
# model = pickle.load(open(f'./models/{MODEL}','rb'))
# SCALAR= 'transformer.pkl'
# sc = pickle.load(open(f'./models/{SCALAR}','rb'))
# UPLOAD_FOLDER = 'uploads'


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #in1 = request.form.get('created_date')
    in1="2020-12-02 17:40:53.232025"
    in2 = request.form.get('name')
    in3 = request.form.get('schema_name')
    in4 = request.form.get('asset_contract_type')
    in5 = request.form.get('created_date')
    in6 = request.form.get('slug')
    in7 = request.form.get('safelist_request_status')
    in8 = request.form.get('address')
    in9 = request.form.get('is_presale')
    # in10 = request.form.get('total_supply')
    in10 = 0.0
    in11 = request.form.get('seller_fee_basis_points')
    in12 = request.form.get('set_contract_dev_seller_fee_basis_points')
    # in13 = request.form.get('opensea_seller_fee_basis_points')
    in13=250
    # in14 = request.form.get('featured')
    in14= False
    in15 = request.form.get('collection_dev_seller_fee_basis_points')
    # in16 = request.form.get('opensea_seller_fee_basis_points')
    in16 = 250

    # float_features = [float(x) for x in request.form.values()]
    # final_features = [np.array(float_features)]
    in10=float(in10)
    in11=int(in11)
    in12=int(in12)
    in13=int(in13)
    in15=int(in15)
    in16=int(in16)


    in9 = in9 == "True"
    # in14 = in4 == "True"
    prediction = predictNFT(in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16)

    # prediction = model.predict( x.transform(final_features))

    if prediction == 1:
        pred = " Your NFT will be sold"
    elif prediction == 0:
        pred = "Your NFT will not be sold , Don't spend your money and your time."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))
#
# @app.route('/download',methods=['POST','GET'])
# def upload():
#     try:
#         if request.method == 'POST':
#         # Get the file from post request
#             f = request.files['file']
#
#             # Save the file to ./uploads
#             basepath = os.path.dirname(__file__)
#             file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#             f.save(file_path)
#             test_df = pd.read_csv(file_path)
#             pred= model.predict(sc.transform(test_df))
#             test_df['pred']= pred
#             #checking for previous prediction file
#             if path.exists('./downloads/prediction.csv'):
#                 os.remove('./downloads/prediction.csv')
#                 print('previous prediction file removed from path')
#             else:
#                 pass
#             test_df.to_csv('./downloads/prediction.csv')
#
#             return send_from_directory('./downloads/','prediction.csv',as_attachment=True,  cache_timeout=0)
#     finally:
#         upload_file = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#         time.sleep(2)
#         os.remove(upload_file)
#         print('clean up done')

if __name__ == "__main__":
    app.run(debug=True)
