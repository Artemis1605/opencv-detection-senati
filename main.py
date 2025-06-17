import cv2
import numpy as np

def main():
    """Detección básica con OpenCV - SENATI"""
    print("🎥 Iniciando detección OpenCV...")
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se puede acceder a la cámara")
        return
    
    print("✅ Cámara inicializada")
    print("Controles:")
    print("- Presiona 'q' para salir")
    print("- Presiona 'c' para activar detección Canny")
    
    canny_mode = False
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error al capturar frame")
            break
        
        # Procesar según modo
        if canny_mode:
            # Modo Canny - Detección de bordes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Mostrar en color
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(processed, "MODO: Deteccion Canny", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Modo normal - Detección de contornos
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 80)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Dibujar contornos
            processed = frame.copy()
            cv2.drawContours(processed, contours, -1, (0, 255, 0), 2)
            
            # Información
            cv2.putText(processed, f"Contornos: {len(contours)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Mostrar controles
        cv2.putText(processed, "Presiona 'c' cambiar modo, 'q' salir", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Agregar contador de frames
        cv2.putText(processed, f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", 
                   (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Mostrar frame
        cv2.imshow('OpenCV Detection - SENATI', processed)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canny_mode = not canny_mode
            print(f"🔄 Modo Canny: {'ON' if canny_mode else 'OFF'}")
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Aplicación cerrada")

if __name__ == "__main__":
    main()
