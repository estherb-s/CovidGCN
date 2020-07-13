import numpy as np
import pandas as pd
import numpy as np
from rdkit.Chem import PandasTools
from rdkit import Chem

def load_and_prepCSV(path):
    df = pd.read_csv(path, header=0)
    df = df.iloc[4:]
    df = df.reset_index() # Remove unnecessary first 4 rows then reset index
    df = df[['PUBCHEM_SID', 'PubChem Standard Value']] # select only necessary columns
    df2 = df.apply(pd.to_numeric) # Set object as float
    # Assign binary value for classification
    df2['Active'] = df2.apply(lambda x: 1 if x['PubChem Standard Value'] <= 5.0 else 0, axis=1)
    df2 = df2.dropna() 
    return df2

def load_and_prepSDF(path):
    SDFFile = path
    pubchem = PandasTools.LoadSDF(SDFFile)
    # Generate SMILES from mol column
    pubchem['SMILES'] = pubchem.apply(lambda x: Chem.MolToSmiles(x['ROMol']), axis=1)
    pubchem = pubchem[['PUBCHEM_SUBSTANCE_ID','SMILES']]
    return pubchem

def finalise_dataset(df, pubchem):
    df = df.join(pubchem) # join the sdf file with csv file
    # save as csv file
    # df.to_csv(r'data\training2.csv', sep='\t')
    return df


def main():
    files = {
        "AID_1409607_datatable_all.csv" : "1817399866942882151.sdf",
        "AID_255062_datatable_all.csv" : "1126838623513267552.sdf",
        "AID_1143407_datatable_all.csv" : "3171584971049892057.sdf",
        "AID_1401288_datatable_all.csv" : "3863973400097120853.sdf",
        "AID_1303367_datatable_all.csv" : "851071281757293998.sdf",
        "AID_1456622_datatable_all.csv" : "2003738735470513881.sdf", 
        "AID_642211_datatable_all.csv" : "2651181772145370329.sdf",
        "AID_1152078_datatable_all.csv" : "2613963822972557379.sdf",
        "AID_1294180_datatable_all.csv" : "3634302240315407858.sdf"
    }

    training_dataset = []
    for csv, sdf in files.item():
        csv_file = f"data/{csv}"
        sdf_file = f"data/{sdf}"
        df = load_and_prepCSV(csv_file)
        pc = load_and_prepSDF(sdf_file)
        training = finalise_dataset(df, pc)
        print(training.groupby('Active').count())
        # store dataframes in a list
        training_dataset.append(training)
    
    training_dataset = pd.concat(training_dataset)
    # # final.head()
    # print(final)

    # print("Done")
if __name__ == "__main__":
    main()