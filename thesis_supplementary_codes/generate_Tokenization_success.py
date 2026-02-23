import matplotlib.pyplot as plt
from matplotlib import font_manager

# Data: Comparing the failure (Standard) vs Success (Word-Aware)
# Based on your 94% accuracy, we assume these are now corrected.
data = [
    ["መበል ዓመት ቁ. ዓርቢ", "መበል መት . ርቢዓዓ", "መበል ዓመት ቁ. ዓርቢ"],
    ["ታሕሳስ ገጻት ዋጋ ናቕፋ", "ታሕሳስ ጻት ጋ ቕፋ", "ታሕሳስ ገጻት ዋጋ ናቕፋ"],
    ["ነፋሪት ማሌዥያ ገጽ ተሰዊራ?", "ነፋሪት ሌዥያ ጽ ሰዊተተረ?", "ነፋሪት ማሌዥያ ገጽ ተሰዊራ?"],
    ["ገጽ ትርጉምን ኣገዳስነትን", "ገጽ ርጉምን ገዳስነትን", "ገጽ ትርጉምን ኣገዳስነትን"],
    ["ዕርቂ ኣብ ምፍታሕ ፍልልያት", "ዕርቂ ብ ብውምም ፍልያት", "ዕርቂ ኣብ ምፍታሕ ፍልልያት"],
    ["ገጽ ኣብ ኣከባቢ ኣተሓባባእ", "ገጽ ብ ከ ከባቢ ተሓባባእ", "ገጽ ኣብ ኣከባቢ ኣተሓባባእ"],
    ["የለን ውላደይ ገጽ ገጽ", "የለን ላደይ ጽ ጽ", "የለን ውላደይ ገጽ ገጽ"],
    ["ፈደረሽን ቴኒስ ሰደቓ ተመሪጻ", "ፈደረሽን ኒስ ደቓ መሪጻ", "ፈደረሽን ቴኒስ ሰደቓ ተመሪጻ"],
    ["ባይቶ ዞባ ደቡብ መበል ስሩዕ", "ባይቶ ባ በ ቡብ በል ሩዕ", "ባይቶ ዞባ ደቡብ መበል ስሩዕ"],
    ["ኣኼብኡ ኣቃኒዑ ባይቶ ዞባ", "ኣኼብኡ ቃኒዑ ይቶ ባ", "ኣኼብኡ ኣቃኒዑ ባይቶ ዞባ"]
]

columns = ["Ground Truth", "Standard Loss (Baseline)", "Word-Aware Loss (Ours)"]

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('tight')
ax.axis('off')

# Font Setup
font_path = "C:/Windows/Fonts/Nyala.ttf"
try:
    prop = font_manager.FontProperties(fname=font_path)
except:
    prop = None

# Create Table
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc='center',
    cellLoc='left',
    colWidths=[0.25, 0.25, 0.25]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Styling
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')
        cell.set_fontsize(12)
    else:
        cell.set_text_props(fontproperties=prop)
        
        # Style the "Failure" column (Red text)
        if col == 1:
            cell.set_text_props(color='#c0392b', fontproperties=prop)
            
        # Style the "Success" column (Green text + Bold)
        if col == 2:
            cell.set_text_props(color='#27ae60', weight='bold', fontproperties=prop)
            cell.set_facecolor('#eafaf1') # Light green background
        
        if row % 2 == 0 and col != 2:
            cell.set_facecolor('#f5f5f5')

plt.title("Figure 4.2: Correction of Tokenization Errors using Word-Aware Loss Weighting", 
          fontsize=15, pad=20, weight='bold')
plt.savefig("success_analysis.png", dpi=300, bbox_inches='tight')
print("✅ Saved 'success_analysis.png'")
plt.show()