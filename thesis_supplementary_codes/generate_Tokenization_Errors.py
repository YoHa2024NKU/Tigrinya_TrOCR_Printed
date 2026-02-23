import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# 1. SETUP DATA
data = [
    ["መበል ዓመት ቁ. ዓርቢ", "መበል መት . ርቢዓዓ", "First char after space missing"],
    ["ገጽ ትርጉምን ኣገዳስነትን", "ገጽ ርጉምን ገዳስነትን", "Missing 'ት', 'ኣ'"],
    ["ገጽ ኣብ ኣከባቢ ኣተሓባባእ", "ገጽ ብ ከ ከባቢ ተሓባባእ", "Missing 'ኣ' (multiple)"],
    ["ባይቶ ዞባ ደቡብ መበል ስሩዕ", "ባይቶ ባ በ ቡብ በል ሩዕ", "Severe token confusion"],
    ["ኣኼብኡ ኣቃኒዑ ባይቶ ዞባ", "ኣኼብኡ ቃኒዑ ይቶ ባ", "Missing 'ኣ', 'ባ', 'ዞ'"]
]

columns = ["Ground Truth (GT)", "Prediction (Standard Loss)", "Error Analysis"]

# 2. SETUP PLOT
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# 3. CONFIGURE FONT 
# Windows usually has 'Nyala'. If on Linux/Mac, use 'Noto Sans Ethiopic' or 'Abyssinica SIL'
font_path = "C:/Windows/Fonts/Nyala.ttf" 
try:
    prop = font_manager.FontProperties(fname=font_path)
    print(f"✅ Loaded font: {prop.get_name()}")
except:
    print("⚠️ Font not found. Using default (Ethiopic might not render).")
    prop = None

# 4. CREATE TABLE
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc='center',
    cellLoc='left',
    colWidths=[0.3, 0.3, 0.4]
)

# 5. STYLE TABLE
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2) # Adjust row height

# Style Headers
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e') # Academic Blue header
        cell.set_fontsize(12)
    else:
        # Set Ethiopic font for data rows
        cell.set_text_props(fontproperties=prop)
        # Highlight errors in the prediction column (Column 1)
        if col == 1:
            cell.set_text_props(color='#c0392b', fontproperties=prop) # Red text for errors
        
        # Alternate row colors
        if row % 2 == 0:
            cell.set_facecolor('#f5f5f5')

# 6. SAVE
plt.title("Figure 3.4: Systematic Tokenization Errors (Character Elision) Before Custom Weighting", 
          fontsize=14, pad=20, weight='bold')
plt.savefig("tokenizer_error_analysis.png", dpi=300, bbox_inches='tight')
print("✅ Image saved as 'tokenizer_error_analysis.png'")
plt.show()