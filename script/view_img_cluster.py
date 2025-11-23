import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os

def debug_clustering_on_image(image_path, xml_path, output_debug_path="debug_clustering.jpg"):
    """
    Fonction de debug visuel : montre comment le clustering sépare tes objets
    """
    # Charger image
    img = cv2.imread(image_path)
    if img is None:
        print("Image non trouvée")
        return
    h, w = img.shape[:2]
    debug = img.copy()

    # Lire les bbox
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    labels = []
    centroids = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        b = obj.find("bndbox")
        x1 = int(b.find("xmin").text)
        y1 = int(b.find("ymin").text)
        x2 = int(b.find("xmax").text)
        y2 = int(b.find("ymax").text)
        bboxes.append([x1, y1, x2, y2])
        labels.append(name)

        # Centroïde
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroids.append([cx, cy])

        # Dessiner bbox originale en blanc
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.putText(debug, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if len(centroids) < 2:
        print("Moins de 2 objets → pas de clustering")
        cv2.imwrite(output_debug_path, debug)
        return

    centroids = np.array(centroids)

    # === CLUSTERING VERTICAL (haut / bas) ===
    y_coords = centroids[:, 1] / h
    median_y = np.median(y_coords)
    group_top = y_coords < median_y
    group_bottom = y_coords >= median_y

    gap_y = np.min(y_coords[group_bottom]) - np.max(y_coords[group_top]) if np.any(group_top) and np.any(group_bottom) else 0

    print(f"Séparation verticale : gap = {gap_y:.3f} → {'OUI' if gap_y > 0.25 else 'NON'}")

    # === CLUSTERING HORIZONTAL (gauche / droite) ===
    x_coords = centroids[:, 0] / w
    median_x = np.median(x_coords)
    group_left = x_coords < median_x
    group_right = x_coords >= median_x

    gap_x = np.min(x_coords[group_right]) - np.max(x_coords[group_left]) if np.any(group_left) and np.any(group_right) else 0

    print(f"Séparation horizontale : gap = {gap_x:.3f} → {'OUI' if gap_x > 0.25 else 'NON'}")

    # === DESSINER LES CLUSTERS ===
    colors = {
        'top': (0, 255, 0),      # vert
        'bottom': (0, 0, 255),   # rouge
        'left': (255, 0, 0),     # bleu
        'right': (0, 255, 255)   # jaune
    }

    for i, (cx, cy) in enumerate(centroids):
        x1, y1, x2, y2 = bboxes[i]
        if gap_y > 0.25:
            color = colors['top'] if group_top[i] else colors['bottom']
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 4)
            cv2.circle(debug, (cx, cy), 15, color, -1)

        if gap_x > 0.25:
            color = colors['left'] if group_left[i] else colors['right']
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 4)
            cv2.circle(debug, (cx, cy), 20, color, 5)

    # === DESSINER LES ZONES DE CROP PROPOSÉES ===
    def draw_crop_zone(bbox_group, color, label):
        if len(bbox_group) == 0: return
        xs = [b[0] for b in bbox_group] + [b[2] for b in bbox_group]
        ys = [b[1] for b in bbox_group] + [b[3] for b in bbox_group]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        obj_w, obj_h = x2 - x1, y2 - y1
        crop_w = max(obj_w * 1.8, 600)
        crop_h = max(obj_h * 1.8, 600)
        cx1 = int(center_x - crop_w // 2)
        cy1 = int(center_y - crop_h // 2)
        cx2 = int(center_x + crop_w // 2)
        cy2 = int(center_y + crop_h // 2)
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(w, cx2), min(h, cy2)
        cv2.rectangle(debug, (cx1, cy1), (cx2, cy2), color, 6)
        cv2.putText(debug, label, (cx1, cy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

    if gap_y > 0.25:
        top_bboxes = [bboxes[i] for i in range(len(bboxes)) if group_top[i]]
        bot_bboxes = [bboxes[i] for i in range(len(bboxes)) if group_bottom[i]]
        draw_crop_zone(top_bboxes, colors['top'], "TOP")
        draw_crop_zone(bot_bboxes, colors['bottom'], "BOTTOM")

    if gap_x > 0.25:
        left_bboxes = [bboxes[i] for i in range(len(bboxes)) if group_left[i]]
        right_bboxes = [bboxes[i] for i in range(len(bboxes)) if group_right[i]]
        draw_crop_zone(left_bboxes, colors['left'], "LEFT")
        draw_crop_zone(right_bboxes, colors['right'], "RIGHT")

    # Légende
    cv2.putText(debug, f"Vertical split: {'YES' if gap_y > 0.25 else 'NO'}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)
    cv2.putText(debug, f"Horizontal split: {'YES' if gap_x > 0.25 else 'NO'}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

    cv2.imwrite(output_debug_path, debug)
    print(f"Debug image sauvegardée → {output_debug_path}")
    cv2.imshow("Clustering Debug - Appuie sur une touche pour fermer", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==================== UTILISATION RAPIDE ====================
if __name__ == "__main__":
    # Change juste ces deux chemins avec une de tes images problématiques
    image_path = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc\test\images\ffdb6560-IMG_R_JA_185.jpg"
    xml_path = image_path.replace("images", "Annotations").replace(".jpg", ".xml")

    debug_clustering_on_image(image_path, xml_path, "debug_clustering.jpg")