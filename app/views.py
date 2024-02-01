from django.shortcuts import render
from django.http import JsonResponse,HttpResponse, Http404, FileResponse
import pubchempy as pcp
import pandas as pd
import numpy as np
import os as os
from keras.models import load_model
from padelpy import from_smiles
from sklearn.preprocessing import OneHotEncoder
import time as tm
import uuid
import pubchempy as pcp
from keras.models import load_model
from django.conf import settings


# Assuming df1 and df2 are defined somewhere in your code
df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
df2 = pd.DataFrame({'col3': [7, 8, 9], 'col4': [10, 11, 12]})

# Load the model
print(settings)
model = load_model("app/static/Model.hdf5", compile=True)


def myfirst(request):
    return render(request, 'myfirst.html')


def processByID(drug_id, excipient_id):
    try:
        CID_D = pcp.Compound.from_cid(str(drug_id))
        CID_E = pcp.Compound.from_cid(str(excipient_id))
    except pcp.PubChemHTTPError:
        return {"error": "Invalid PubChem Compound ID"}

    FPD = CID_D.cactvs_fingerprint
    FPE = CID_E.cactvs_fingerprint

    List1 = list(FPD)
    List2 = list(FPE)
    List = List1 + List2

    t = pd.DataFrame(np.array(List).reshape(-1, len(List)))
    dataset1 = t.values
    X_Predict = dataset1[:, 0:1762].astype(int)

    t1 = model.predict(X_Predict)
    probability_compatible = t1[0][0] * 100

    output_string = f"Compatible. Probability: {probability_compatible:.2f}%"

    return {"output": output_string}

def process_by_id(request):
    if request.method == 'POST':
        form_type = request.POST.get('form_type')

        if form_type == 'ID':
            drug_id = request.POST.get('one')
            excipient_id = request.POST.get('two')

            print("Drug ID:", drug_id)
            print("Excipient ID:", excipient_id)

            result_data = processByID(drug_id, excipient_id)

            return JsonResponse(result_data)

        return JsonResponse({"error": "Invalid form type"})

    return JsonResponse({"error": "Invalid request method"})

def processBySmile(drug_smile, excipient_smile):
    try:
        CID_D = from_smiles(drug_smile, fingerprints=True, descriptors=False)
        CID_E = from_smiles(excipient_smile, fingerprints=True, descriptors=False)
    except Exception as e:
        return {"error": str(e)}

    FPD = list(CID_D.values())
    FPE = list(CID_E.values())

    List = FPD + FPE

    t = pd.DataFrame(np.array(List).reshape(-1, len(List)))
    dataset1 = t.values
    X_Predict = dataset1[:, 0:1762].astype(int)
    # Use model.predict_proba instead of model.predict
    t1 = model.predict(X_Predict)
    

    # Add debug prints
    print("t1 values:", t1)

    probability_compatible = t1[0][1] * 100
 # Use the correct index for probability of compatibility
    if probability_compatible >= 50.0:
        output_string = f"Compatible. Probability: {probability_compatible:.2f}%"
    else:
        output_string = f"Incompatible. Probability: {probability_compatible:.2f}%"

    print("Output string:", output_string)

    return {"output": output_string}

def process_by_smile(request):
    if request.method == 'POST':
        form_type = request.POST.get('form_type')

        if form_type == 'SMILE':
            drug_smile = request.POST.get('one1')
            excipient_smile = request.POST.get('two2')

            print("Drug SMILE:", drug_smile)
            print("Excipient SMILE:", excipient_smile)

            result_data = processBySmile(drug_smile, excipient_smile)

            return JsonResponse(result_data)

        return JsonResponse({"error": "Invalid form type"})

    return JsonResponse({"error": "Invalid request method"})

def process_by_excel(request):
    if request.method == 'POST' and request.FILES:
        uploaded_file = request.FILES['file']
        newFilePath = 'app/static/uploads/'+uploaded_file.name
        with open(newFilePath, 'wb+') as destination: 
            for chunk in uploaded_file.chunks(): 
                destination.write(chunk) 
       
        df = pd.read_excel(newFilePath)

        # Define a function to calculate compatibility percentage
        def calculate_percentage(drug_id, excipient_id):
            try:
                CID_D = pcp.Compound.from_cid(str(drug_id))
                CID_E = pcp.Compound.from_cid(str(excipient_id))
            except pcp.PubChemHTTPError:
                return {"error": "Invalid PubChem Compound ID"}

            FPD = CID_D.cactvs_fingerprint
            FPE = CID_E.cactvs_fingerprint

            List1 = list(FPD)
            List2 = list(FPE)
            List = List1 + List2

            t = pd.DataFrame(np.array(List).reshape(-1, len(List)))
            dataset1 = t.values
            X_Predict = dataset1[:, 0:1762].astype(int)

            t1 = model.predict(X_Predict)
            probability_compatible = t1[0][0] * 100

            output_string = f"Compatible. Probability: {probability_compatible:.2f}%"
            
            return {"drug_id": drug_id, "excipient_id": excipient_id, "output": output_string, "percentage": probability_compatible}

        # Process drugs and excipients
        result_dfs = []
        for index, row in df.iterrows():
            drug_id = row['Drug_CID']
            excipient_id = row['Excipient_CID']

            compatibility_result = calculate_percentage(drug_id, excipient_id)
            result_dfs.append(compatibility_result)

        # Concatenate all result dataframes
        result_df = pd.DataFrame(result_dfs)

        # Save the processed data to a new Excel file
        fName = str(uuid.uuid4())+".xlsx"
        result_df.to_excel(f"app/static/results/{fName}")

        os.remove(newFilePath)
        # Return the results as a JSON response
        return JsonResponse({"success": 'true', "path": f"results/{fName}", "results": result_df.to_dict(orient='records')})
    
    return JsonResponse({'error': 'Invalid request'})

def download_excel(request):
    if request.method == 'GET':
        filePath = request.GET.get('path')
        response = FileResponse(open(f"app/static/{filePath}", 'rb'), as_attachment=True)
        return response
    raise Http404