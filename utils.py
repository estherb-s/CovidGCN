import numpy as np
import pandas as pd
import numpy as np
from rdkit.Chem import PandasTools
from rdkit import Chem

def load_and_prepCSV(path):
    df = pd.read_csv(path, header=0)
    df = df.iloc[4:]
    df = df.reset_index()
    df = df[['PUBCHEM_SID', 'PubChem Standard Value']]
    df2 = df.apply(pd.to_numeric)
    df2['Active'] = df2.apply(lambda x: 1 if x['PubChem Standard Value'] <= 5.0 else 0, axis=1)
    df2 = df2.dropna() # Keep or not - test
    return df2

def load_and_prepSDF(path):
    SDFFile = path
    pubchem = PandasTools.LoadSDF(SDFFile)
    pubchem['SMILES'] = pubchem.apply(lambda x: Chem.MolToSmiles(x['ROMol']), axis=1)
    pubchem = pubchem[['PUBCHEM_SUBSTANCE_ID','SMILES']]
    return pubchem

def finalise_dataset(df, pubchem):
    df = df.join(pubchem)
    # df.to_csv(r'data\training2.csv', sep='\t')
    return df


def main():

    csv = f"data/AID_255062_datatable_all.csv"
    pubchem = f"data/1126838623513267552.sdf"
    df = load_and_prepCSV(csv)
    pc = load_and_prepSDF(pubchem)
    final = finalise_dataset(df, pc)
    # final.head()
    print(final)


    # for file,link in files.items():
    #     path = f"data/{file}.pkl"
        # if not exists(path):
        #     download_tsv(link, file+".tsv")
        #     print("Converting to pickle")
        #     df = load_and_fillna(path.replace("pkl", "tsv"))
        #     save_to_pickle(df, file+".pkl")
        #     save_to_pickle(df, "std_"+file+".pkl", standardise=True)
        #     cleanup_tsv()

    # print("Done")
if __name__ == "__main__":
    main()