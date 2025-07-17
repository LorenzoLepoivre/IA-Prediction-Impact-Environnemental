import pandas as pd
import pickle

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print("Modèle chargé.")
    return model

def predict_single(model, input_data: dict):
    df = pd.DataFrame([input_data])
    
    # Ici tu peux vérifier que toutes les colonnes attendues sont là :
    expected_cols = model.feature_names_in_  # si sklearn >=1.0 sinon hardcode ta liste
    print(f"Nombre de colonnes attendues : {len(expected_cols)}")

    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la prédiction : {missing}")
    
    df = df[expected_cols]  # ordre exact

    pred = model.predict(df)
    return pred[0]

if __name__ == "__main__":
    model = load_model()
    
    input_data = {
    "Code AGB": 11084,
    "Groupe d'aliment": "aides culinaires et ingrédients divers",
    "Sous-groupe d'aliment": "algues",
    "Nom du Produit en Français": "Agar (algue), cru",
    "LCI Name": "Seaweed, agar, raw",
    "Livraison": "Ambiant (long)",
    "Approche emballage ": "PACK PROXY",
    "Préparation": "Pas de préparation",
    "Rayonnements ionisants": 2.77,
    "Formation photochimique d'ozone": 1.55,
    "Effets toxicologiques sur la santé humaine\xa0: substances non-cancérogènes": 11.8,
    "Effets toxicologiques sur la santé humaine\xa0: substances cancérogènes": 6.27e-7,
    "Eutrophisation eaux douces": 11.2,
    "Eutrophisation marine": 0.0514,
    "Écotoxicité pour écosystèmes aquatiques d'eau douce": 8.62e-7,
    "Utilisation du sol": 9.85e-8,
    "Épuisement des ressources énergétiques": 6.93e-9,
    "Épuisement des ressources minéraux": 111,
    "Changement climatique - émissions fossiles": 0.00215,
    "Changement climatique - émissions liées au changement d'affectation des sols": 15
}


    prediction = predict_single(model, input_data)
    print(f"Prédiction : {prediction}")
