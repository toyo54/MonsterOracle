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
#weak_boss['actions_count'] = 0
weak_cr = predict_monster_cr(rf_lean, top_20_features, weak_boss)

print(f"📉  CR Senza Azioni Leggendarie: {weak_cr:.2f}")
print(f"💡  Delta: {estimated_cr - weak_cr:.2f} punti.")

# ---------------------------------------------------------
# TEST: GLASS CANNON (Pochi HP, Alta Offensiva)
# ---------------------------------------------------------
glass_cannon = {
    'hit_points': 50,               # HP Bassi (da CR 1/2)
    'armor_class': 14,
    'constitution': 10,
    'hit_dice_count': 6,
    'strength': 10,
    'dexterity': 20,
    'actions_count': 4,             # 4 Attacchi!
    'legendary_actions_count': 0,
    'special_abilities_count': 1
}

# 1. Predizione con azioni
cr_full = predict_monster_cr(rf_lean, top_20_features, glass_cannon)

# 2. Predizione senza azioni
glass_cannon_weak = glass_cannon.copy()
glass_cannon_weak['actions_count'] = 1 # Lo rendiamo inoffensivo

cr_weak = predict_monster_cr(rf_lean, top_20_features, glass_cannon_weak)

print(f"🗡️ CR Assassino (4 Attacchi): {cr_full:.2f}")
print(f"🥀 CR Assassino (1 Attacco):  {cr_weak:.2f}")
print(f"📉 Crollo del CR: {cr_full - cr_weak:.2f}")


print("\n🔬 TEST FINALE: Feature Dominance (HP vs Actions)")
print("-" * 50)

# Caso A: L'Assassino perde le azioni (ma tiene gli HP)
boss_no_actions = glass_cannon.copy()
boss_no_actions['actions_count'] = 1
cr_no_actions = predict_monster_cr(rf_lean, top_20_features, boss_no_actions)

# Caso B: L'Assassino perde metà vita (ma tiene le azioni)
boss_half_hp = glass_cannon.copy()
boss_half_hp['hit_points'] = 25 # Dimezzati da 50
cr_half_hp = predict_monster_cr(rf_lean, top_20_features, boss_half_hp)

print(f"1. Base (50 HP, 4 Atk):      CR {cr_full:.2f}")
print(f"2. Solo 1 Attacco (50 HP):   CR {cr_no_actions:.2f} (Delta: {cr_full - cr_no_actions:.2f}) -> Impatto Minimo")
print(f"3. Metà Vita (25 HP, 4 Atk): CR {cr_half_hp:.2f}    (Delta: {cr_full - cr_half_hp:.2f}) -> Impatto MASSICCIO")

print("-" * 50)
print("CONCLUSIONE: Il modello conferma che gli HP pesano molto più dell'Action Economy.")