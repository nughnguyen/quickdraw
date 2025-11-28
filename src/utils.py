
import cv2
import numpy as np
from sklearn import metrics
import torch


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def get_images(path, classes):
    """Load class images with error handling"""
    images = []
    for item in classes:
        try:
            img = cv2.imread(f"{path}/{item}.png", cv2.IMREAD_UNCHANGED)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image for {item}: {e}")
            images.append(None)
    return images


def get_overlay(bg_image, fg_image, sizes=(40, 40)):
    """Overlay foreground image on background with transparency"""
    if fg_image is None:
        return bg_image
    
    try:
        fg_image = cv2.resize(fg_image, sizes)
        fg_mask = fg_image[:, :, 3:]
        fg_image = fg_image[:, :, :3]
        bg_mask = 255 - fg_mask
        bg_image = bg_image/255
        fg_image = fg_image/255
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)/255
        bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)/255
        image = cv2.addWeighted(bg_image*bg_mask, 255, fg_image*fg_mask, 255, 0.).astype(np.uint8)
        return image
    except Exception as e:
        print(f"Warning: Overlay failed: {e}")
        return bg_image


def get_confidence_score(model, image_tensor):
    """Calculate confidence score for prediction"""
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits[0], dim=0)
        confidence = torch.max(probabilities).item()
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, confidence


def preprocess_drawing(canvas, min_area=3000):
    """Preprocess drawn image for recognition"""
    canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Apply blur
    median = cv2.medianBlur(canvas_gs, 9)
    gaussian = cv2.GaussianBlur(median, (5, 5), 0)
    
    # Otsu's thresholding
    _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return None, "No drawing detected"
    
    # Get largest contour
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    # Check minimum area
    if cv2.contourArea(contour) < min_area:
        return None, "Drawing too small"
    
    # Extract and resize
    x, y, w, h = cv2.boundingRect(contour)
    image = canvas_gs[y:y + h, x:x + w]
    image = cv2.resize(image, (28, 28))
    image = np.array(image, dtype=np.float32)[None, None, :, :]
    
    return image, "success"
