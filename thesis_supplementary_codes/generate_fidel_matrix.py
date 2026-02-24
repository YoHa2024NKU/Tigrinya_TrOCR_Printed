from PIL import Image, ImageDraw, ImageFont
import os


def generate_full_tigrinya_matrix():
    # =========================================================
    # 1. DATA DEFINITIONS
    # =========================================================

    # --- A. Core Bases (33) ---
    bases_hex = [
        0x1200, 0x1208, 0x1210, 0x1218, 0x1220, 0x1228, 0x1230, 0x1238,
        0x1240, 0x1260, 0x1270, 0x1278, 0x1280, 0x1290, 0x1298, 0x12A0,
        0x12A8, 0x12B8, 0x12C8, 0x12D0, 0x12D8, 0x12E0, 0x12E8, 0x12F0,
        0x1300, 0x1308, 0x1320, 0x1328, 0x1330, 0x1338, 0x1340, 0x1348,
        0x1350,
    ]

    transliterations = [
        "h", "l", "\u1e25", "m", "\u1e61", "r", "s", "\u0161",
        "q", "b", "t", "\u010d", "\u1e2b", "n", "\u00f1", "\u02be",
        "k", "x", "w", "\u02bf", "z", "\u017e", "y", "d",
        "\u01e7", "g", "\u1e6d", "\u010d\u0323", "\u1e57", "\u1e63", "\u1e63\u0301", "f",
        "p"
    ]

    vowels = ["\u00e4", "u", "i", "a", "e", "(\u0259)", "o"]
    headers = ["", "\u00e4/e\n[e]", "u\n[u]", "i\n[i]", "a\n[a]",
               "e\n[e]", "(\u0259)\n[\u0268]", "o\n[o]"]

    # --- B. Labialized Consonants (4 groups) ---
    lab_bases_hex = [0x1248, 0x1288, 0x12B0, 0x1310]
    lab_transliterations = ["q\u02b7", "\u1e2b\u02b7", "k\u02b7", "g\u02b7"]
    lab_offsets = {1: 0, 3: 2, 4: 3, 5: 4, 6: 5}

    # --- C. Punctuation ---
    punctuations = [
        ("Word Space", "\u1361", "U+1361"),
        ("Full Stop", "\u1362\u1362", "U+1362"),
        ("Comma", "\u1363", "U+1363"),
        ("Semicolon", "\u1364", "U+1364"),
        ("Colon", "\u1365", "U+1365"),
        ("Preface Colon", "\u1366", "U+1366"),
        ("Question Mark", "\u1367", "U+1367"),
        ("Paragraph Sep.", "\u1368", "U+1368")
    ]

    # =========================================================
    # 2. IMAGE DIMENSIONS & SETUP
    # =========================================================
    cell_width = 85
    cell_height = 70
    cols_per_table = 8
    rows_core = 18  # 1 header + 17 data rows

    table_width = cols_per_table * cell_width
    gap_between_tables = 60

    margin_top = 160
    margin_bottom = 80
    margin_x = 60
    middle_gap_y = 100

    core_height = rows_core * cell_height
    bottom_height = 9 * 60

    img_width = (table_width * 2) + gap_between_tables + (margin_x * 2)
    img_height = margin_top + core_height + middle_gap_y + bottom_height + margin_bottom

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # =========================================================
    # 3. FONTS
    # =========================================================
    font_path_fidel = r"D:\Tigrinya_OCR_Project\nyala.ttf"
    font_path_latin = "C:/Windows/Fonts/times.ttf"

    try:
        fidel_font = ImageFont.truetype(font_path_fidel, 46)
        latin_font = ImageFont.truetype(font_path_latin, 15)
        header_font = ImageFont.truetype(font_path_latin, 26)
        title_font = ImageFont.truetype(font_path_latin, 32)
        subtitle_font = ImageFont.truetype(font_path_latin, 18)
        section_font = ImageFont.truetype(font_path_latin, 20)
    except IOError:
        print("Warning: System fonts not found.")
        fidel_font = latin_font = header_font = title_font = \
            subtitle_font = section_font = ImageFont.load_default()

    # =========================================================
    # 4. DRAW MAIN TITLES
    # =========================================================
    title_text = "The Tigrinya Writing System (Ge'ez Script)"
    subtitle_text = "Core Syllabary (231) | Labio-Velar Forms (20) | Punctuation Marks (8)"  # FIXED: 33x7=231

    draw.text((img_width // 2, 50), title_text, fill="black",
              font=title_font, anchor="mm", align="center")
    draw.text((img_width // 2, 95), subtitle_text, fill="#333333",
              font=subtitle_font, anchor="mm", align="center")

    # =========================================================
    # 5. FUNCTION: DRAW CORE TABLES
    # =========================================================
    def draw_core_table(start_x, start_y, base_start_idx, base_end_idx):
        for row in range(rows_core):
            y = start_y + (row * cell_height)
            for col in range(cols_per_table):
                x = start_x + (col * cell_width)
                draw.rectangle([x, y, x + cell_width, y + cell_height],
                               outline="black", width=1)

                cx = x + (cell_width // 2)
                cy_top = y + (cell_height * 0.35)
                cy_bottom = y + (cell_height * 0.75)

                if row == 0:
                    if col > 0:
                        parts = headers[col].split('\n')
                        draw.text((cx, cy_top), parts[0], fill="black",
                                  font=header_font, anchor="mm")
                        draw.text((cx, cy_bottom), parts[1], fill="#444444",
                                  font=header_font, anchor="mm")
                else:
                    data_idx = base_start_idx + row - 1

                    # FIXED: bounds check to prevent IndexError
                    if data_idx > base_end_idx or data_idx >= len(bases_hex):
                        continue

                    base_trans = transliterations[data_idx]

                    if col == 0:
                        if base_trans not in ['\u02be', '\u02bf']:
                            display_base = base_trans + "a"
                        else:
                            display_base = base_trans + "\u00e4"
                        draw.text((cx, cy_top), display_base, fill="black",
                                  font=header_font, anchor="mm")
                        draw.text((cx, cy_bottom), f"[{base_trans}]",
                                  fill="#444444", font=header_font, anchor="mm")
                    else:
                        vowel_idx = col - 1
                        fidel_char = chr(bases_hex[data_idx] + vowel_idx)
                        trans_text = f"{base_trans}{vowels[vowel_idx]}"

                        draw.text((cx, cy_top), fidel_char, fill="black",
                                  font=fidel_font, anchor="mm")
                        draw.text((cx, cy_bottom), trans_text, fill="#555555",
                                  font=latin_font, anchor="mm")

    # Draw Left & Right Core Tables
    left_x = margin_x
    right_x = margin_x + table_width + gap_between_tables
    draw_core_table(left_x, margin_top, 0, 16)     # 17 consonants (h through k)
    draw_core_table(right_x, margin_top, 17, 32)   # 16 consonants (x through p)

    # =========================================================
    # 6. DRAW BOTTOM LEFT: LABIALIZED FORMS (4 groups)
    # =========================================================
    bottom_y = margin_top + core_height + middle_gap_y
    draw.text((left_x + table_width // 2, bottom_y - 25),
              "Labialized (Labio-Velar) Forms", fill="black",
              font=section_font, anchor="mm")

    lab_rows = 1 + len(lab_bases_hex)
    for row in range(lab_rows):
        y = bottom_y + (row * cell_height)
        for col in range(cols_per_table):
            x = left_x + (col * cell_width)
            draw.rectangle([x, y, x + cell_width, y + cell_height],
                           outline="black", width=1)
            cx = x + (cell_width // 2)
            cy_top = y + (cell_height * 0.35)
            cy_bottom = y + (cell_height * 0.75)

            if row == 0:
                if col > 0:
                    parts = headers[col].split('\n')
                    draw.text((cx, cy_top), parts[0], fill="black",
                              font=header_font, anchor="mm")
                    draw.text((cx, cy_bottom), parts[1], fill="#444444",
                              font=header_font, anchor="mm")
            else:
                base_idx = row - 1
                base_trans = lab_transliterations[base_idx]
                if col == 0:
                    draw.text((cx, cy_top),
                              base_trans.replace('\u02b7', 'wa'),
                              fill="black", font=header_font, anchor="mm")
                    draw.text((cx, cy_bottom), f"[{base_trans}]",
                              fill="#444444", font=header_font, anchor="mm")
                else:
                    if col in lab_offsets:
                        offset = lab_offsets[col]
                        fidel_char = chr(lab_bases_hex[base_idx] + offset)
                        trans_text = f"{base_trans}{vowels[col - 1]}"
                        draw.text((cx, cy_top), fidel_char, fill="black",
                                  font=fidel_font, anchor="mm")
                        draw.text((cx, cy_bottom), trans_text, fill="#555555",
                                  font=latin_font, anchor="mm")
                    else:
                        draw.text((cx, y + (cell_height // 2)), "\u2014",
                                  fill="#aaaaaa", font=fidel_font, anchor="mm")

    # =========================================================
    # 7. DRAW BOTTOM RIGHT: PUNCTUATION
    # =========================================================
    draw.text((right_x + table_width // 2, bottom_y - 25),
              "Tigrinya Punctuation Marks", fill="black",
              font=section_font, anchor="mm")

    punc_col_widths = [260, 160, 260]
    punc_row_height = 60

    for row in range(9):
        y = bottom_y + (row * punc_row_height)
        current_x = right_x

        for col, w in enumerate(punc_col_widths):
            draw.rectangle([current_x, y, current_x + w, y + punc_row_height],
                           outline="black", width=1)
            cx, cy = current_x + (w // 2), y + (punc_row_height // 2)

            if row == 0:
                headers_punc = ["Name", "Symbol", "Unicode"]
                draw.text((cx, cy), headers_punc[col], fill="black",
                          font=header_font, anchor="mm")
            else:
                punc_data = punctuations[row - 1]
                if col == 1:
                    draw.text((cx, cy), punc_data[col], fill="black",
                              font=fidel_font, anchor="mm")
                else:
                    draw.text((cx, cy), punc_data[col], fill="#333333",
                              font=header_font, anchor="mm")

            current_x += w

    # =========================================================
    # 8. SAVE & DISPLAY
    # =========================================================
    output_filename = "tigrinya_full_writing_system.png"
    img.save(output_filename, dpi=(300, 300))
    print(f"Successfully generated and saved '{output_filename}'")

    try:
        os.startfile(output_filename)
    except AttributeError:
        pass


if __name__ == "__main__":
    generate_full_tigrinya_matrix()