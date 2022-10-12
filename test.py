import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

def test():
    file_path = "classset.txt"
    
    with open(file_path, 'r', encoding="UTF-8") as f:
        actions = f.read().splitlines()
        #actions.append(' ')
        print(actions)
    
    seq_length = 30

    model = load_model('models/model2_1.0.h5')

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1080) # set video widht
    cap.set(4, 720) # set video height

    seq = []

    while cap.isOpened():
        ret, img = cap.read()
        img0 = img.copy()
        
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                # 인퍼런스한 결과를 뽑아냄
                y_pred = model.predict(input_data).squeeze()

                # argmax로 인덱스 뽑아냄
                # 그것의 confidence를 뽑아냄
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                # confidence가 90% 이하면 확실하지 않다 판단해 넘김
                if conf < 0.9:
                    continue
                
                # 90% 이상이면 action 라벨명 추출
                action = actions[i_pred]
                
                # 출력
                img_pil=Image.fromarray(img)

                # PIL 이미지에 한글 입력
                draw = ImageDraw.Draw(img_pil)
                draw.text((40, 450),  f"{action}", font=ImageFont.truetype("./malgun.ttf", 36), fill=(255,255,255))

                # PIL 이미지 -> cv2 Mat 타입으로 변경
                img = np.array(img_pil)
                
                
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break
    

# test()