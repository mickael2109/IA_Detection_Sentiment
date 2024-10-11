from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import cv2
import numpy as np
import joblib
import os

@csrf_exempt
def predict_image_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        # Sauvegarder l'image temporairement
        file_name = default_storage.save(image_file.name, image_file)
        file_path = os.path.join(default_storage.location, file_name)

        try:
            # Charger le modèle
            model_path = 'E:/JOBLIB/sentiment/svm_model.pkl'  # ou chemin relatif 'svm_model.pkl'
            model = joblib.load(model_path)

            # Lire et prétraiter l'image
            input_img = cv2.imread(file_path)
            img_rows, img_cols = 48, 48  # Adapter aux dimensions attendues par le modèle
            input_img_resize = cv2.resize(input_img, (img_rows, img_cols))

            # Ajuster si l'image est en niveaux de gris
            if len(input_img_resize.shape) == 2:
                input_img_resize = np.stack((input_img_resize,) * 3, axis=-1)

            input_img_resize = input_img_resize.flatten().reshape(1, -1)

            # Vérifier le nombre de caractéristiques
            if input_img_resize.shape[1] != model.n_features_in_:
                return JsonResponse({'error': f"L'image a {input_img_resize.shape[1]} caractéristiques, mais le modèle en attend {model.n_features_in_}."}, status=400)

            # Faire la prédiction
            prediction = model.predict(input_img_resize)
            response = {'prediction': int(prediction[0])}

        except Exception as e:
            response = {'error': str(e)}
            return JsonResponse(response, status=500)

        finally:
            # Supprimer le fichier temporaire
            default_storage.delete(file_name)

        return JsonResponse(response)

    return JsonResponse({'error': 'Aucune image reçue'}, status=400)
