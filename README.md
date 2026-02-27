import cv2
import numpy as np
import os

# =========================
# 1. DOC ANH CHUAN
# =========================
img_chuan = cv2.imread("chuan.jpg")

if img_chuan is None:
    print("Khong doc duoc anh chuan")
    exit()

img_chuan = cv2.resize(img_chuan, (800,600))
gray_chuan = cv2.cvtColor(img_chuan, cv2.COLOR_BGR2GRAY)

# =========================
# 2. DUYET FILE
# =========================
for ten_file in os.listdir():

    if ten_file == "chuan.jpg" or not ten_file.lower().endswith((".jpg", ".png")):
        continue

    print("Dang kiem tra:", ten_file)

    img_sp = cv2.imread(ten_file)
    if img_sp is None:
        continue

    img_sp = cv2.resize(img_sp, (800, 600))
    gray_sp = cv2.cvtColor(img_sp, cv2.COLOR_BGR2GRAY)

    # =========================
    # 3. SO SANH
    # =========================
    diff = cv2.absdiff(gray_chuan, gray_sp)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_ketqua = img_sp.copy()

    co_loi = False
    so_loi = 0
    so_loi_thieu = 0
    so_loi_sai = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        co_loi = True
        so_loi += 1

        x, y, w, h = cv2.boundingRect(cnt)

        # Lấy vùng tương ứng
        vung_chuan = gray_chuan[y:y+h, x:x+w]
        vung_sp = gray_sp[y:y+h, x:x+w]

        mean_chuan = np.mean(vung_chuan)
        mean_sp = np.mean(vung_sp)

        # =========================
        # PHAN LOAI LOI
        # =========================
        if mean_sp < mean_chuan:
            loai_loi = "Thieu linh kien"
            so_loi_thieu += 1
            mau = (255, 0, 0)  # Xanh duong
        else:
            loai_loi = "Sai linh kien"
            so_loi_sai += 1
            mau = (0, 0, 255)  # Do

        # Ve khung
        cv2.rectangle(img_ketqua, (x, y), (x+w, y+h), mau, 2)

        # Ghi ten loi ngay tren khung
        cv2.putText(img_ketqua, loai_loi, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mau, 2)

    # =========================
    # 4. GHEP ANH (CUA SO 1)
    # =========================
    anh_ghep = np.hstack((img_chuan, img_ketqua))
    cv2.imshow("He thong kiem tra san pham", anh_ghep)

    # =========================
    # 5. CUA SO THONG BAO (CUA SO 2)
    # =========================
    bang_thong_bao = np.zeros((300, 600, 3), dtype=np.uint8)

    if co_loi:
        text1 = "SAN PHAM KHONG DAT"
        text2 = f"Tong loi: {so_loi}"
        text3 = f"Thieu: {so_loi_thieu} | Sai: {so_loi_sai}"
        mau = (0, 0, 255)
    else:
        text1 = "SAN PHAM DAT"
        text2 = "Khong co loi"
        text3 = ""
        mau = (0, 255, 0)

    cv2.putText(bang_thong_bao, text1, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, mau, 3)

    cv2.putText(bang_thong_bao, text2, (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mau, 2)

    cv2.putText(bang_thong_bao, text3, (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mau, 2)

    cv2.imshow("Thong bao ket qua", bang_thong_bao)

    cv2.waitKey(0)

cv2.destroyAllWindows()
