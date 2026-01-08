import cv2

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inférence
    results = model(frame)
    
    # Afficher les résultats
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLO TFLite', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()