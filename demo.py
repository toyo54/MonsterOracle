# ==============================================================================
# 🛠️ SETUP & TRAINING (Esegui questo per creare il modello rf_lean)
# ==============================================================================
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. CARICAMENTO DATI
# ------------------------------------------------------------------------------
df = pd.read_csv("./data/monsters_lean.csv")

# 2. DEFINIZIONE DELLE FEATURE (Top 20)
# ------------------------------------------------------------------------------
# Definiamo manualmente le colonne per essere sicuri che esistano
top_20_features = [
    'hit_points', 'armor_class', 'hit_dice_count', 'constitution',
    'strength', 'dexterity', 'charisma', 'wisdom_save', 'constitution_save',
    'damage_immunities_count', 'damage_resistances_count', 'condition_immunities_count',
    'actions_count', 'legendary_actions_count', 'special_abilities_count',
    'intelligence_save', 'passive_perception', 'charisma_save', 'dexterity_save',
    'skills.stealth'
]

# Controllo esistenza colonne nel CSV (se mancano, le crea a 0)
for col in top_20_features:
    if col not in df.columns:
        print(f"Colonna creata: {col}")
        df[col] = 0

# 3. ADDESTRAMENTO MODELLO
# ------------------------------------------------------------------------------
print("⚙️  Addestramento del modello 'rf_lean' in corso...")

X = df[top_20_features].fillna(0)
y = df['cr'] if 'cr' in df.columns else df.iloc[:, -1] # Fallback se manca 'cr'

# Addestriamo
rf_lean = RandomForestRegressor(n_estimators=100, random_state=42)
rf_lean.fit(X, y)

print("✅  Modello addestrato con successo.")
print("-" * 60)

# ==============================================================================
# 🎲 PREDICTION
# ==============================================================================

# 1. FUNZIONE DI PREDIZIONE
def predict_monster_cr(model, feature_list, monster_data):
    input_df = pd.DataFrame([monster_data])
    input_df = input_df.reindex(columns=feature_list, fill_value=0)
    return model.predict(input_df)[0]

# 2. DEFINIZIONE MOSTRO CUSTOM
print("🛠️  Generazione del mostro 'Lord of Cinder'...")

custom_boss = {
    'hit_points': 200, 'armor_class': 19, 'constitution': 22, 'hit_dice_count': 20,
    'strength': 24, 'dexterity': 14, 'charisma': 20,
    'wisdom_save': 9, 'constitution_save': 12,
    'damage_immunities_count': 2, 'condition_immunities_count': 4, 'damage_resistances_count': 1,
    'actions_count': 3, 'legendary_actions_count': 3, 'special_abilities_count': 2
}

# 3. PREDIZIONE
estimated_cr = predict_monster_cr(rf_lean, top_20_features, custom_boss)

# Report
print(f"🐉  SCHEDA: LORD OF CINDER | HP: {custom_boss['hit_points']} | AC: {custom_boss['armor_class']}")
print(f"🔮  CR PREDETTO: {estimated_cr:.2f}")

# 4. TEST SENSIBILITÀ
weak_boss = custom_boss.copy()
weak_boss['legendary_actions_count'] = 0
weak_cr = predict_monster_cr(rf_lean, top_20_features, weak_boss)

print(f"📉  CR Senza Azioni Leggendarie: {weak_cr:.2f}")
print(f"💡  Delta: {estimated_cr - weak_cr:.2f} punti.")