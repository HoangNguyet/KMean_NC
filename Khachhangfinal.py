# Bốn nhóm khách hàng:
# - Khách hàng chi tiêu thấp: Nhận nhiều chiết khấu nhưng lợi nhuận kh cao
# - Khách hàng chi tiêu cao
# - Khách hàng săn chiết khấu: Mua hàng với số lương lớn nhưng kh mang đến lợi nhuận cho doanh nghiệp
# - Khách hàng trung bình: doanh số và lợi nhuận thấp, không yêu cầu chiết khẩu lớn

#axis=0: Tính từng dòng
#axis =1: Tính theo từng cột
# B1: Thêm các modun cần thiết
import sys
import io

from tkinter import Tk, Label, Entry, LabelFrame

from tkinter import *
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt #tạo đồ thị
import numpy as np # xử lý các mảng, ma trận => dùng trong phép biến đổi
import pandas as pd #thao tác với các tệp
from math import sqrt
from tkinter import *
from tkinter import messagebox
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples # đánh giá mức độ tốt của việc phân cụm
from sklearn.preprocessing import StandardScaler
# Đọc vào file csv
data = pd.read_csv('data_customers_xuly.csv')

# Thống kê các trường mô tả của bộ dữ liệu
# data_customers = data.groupby("Customer ID")[['Sales', 'Quantity', 'Discount','Order ID', 'Profit']]\
# .agg({'Sales':'sum','Quantity':'sum','Discount':'mean','Order ID':'count','Profit':'sum'})\
# .reset_index()

# data_customers.describe()

#chọn các cột là dữ liệu số
numeric_columns = data.select_dtypes(include=[np.number])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaler = scaler.fit_transform(numeric_columns)

X = pd.DataFrame(X_scaler, columns=numeric_columns.columns)

# print('data:\n',data) # in ra tất cả các dữ liệu

# print('X_scaler:\n',X_scaler) # in ra tất cả các dữ liệu

# print('X:\n',X) # in ra tất cả các dữ liệu

# print('column:\n',columns) # in ra tất cả các dữ liệu

# print('data_values:\n',data_values) # in ra tất cả các dữ liệu

# print('numeric:\n',numeric_columns) # in ra tất cả các dữ liệu


#Khởi tạo k cụm
k = 4
#Khởi tạo k tâm ngẫu nhiên từ 
old_centroids = (X.sample(n=k).copy())
print("Tâm cụm khởi tạo: \n", old_centroids)
#Tính khoảng cách giữa 2 điểm

def distance(c1, c2):
    d = sqrt((c1['Sales']-c2['Sales'])**2 + (c1['Quantity']-c2['Quantity'])**2 + (c1['Discount']-c2['Discount'])**2 +(c1['Order ID']-c2['Order ID'])**2 + (c1['Profit']-c2['Profit'])**2 )
    return d

diff = 1
loop = 1
j = 0
loop_max = 100 #Đặt số lần lặp max tránh lặp lâu quá không ra kết quả

while(diff > 1e-5 and loop <= loop_max):
    print("\nLần lặp thứ: ", loop)
    i = 1 # i dùng để đánh số cho các cụm
    for index1, c1 in old_centroids.iterrows(): #Duyệt qua từng tâm cụm, c1 là đại diện cho từng cụm
        ED = [] #Tạo mảng lưu khoảng cách
        for index2, c2 in X.iterrows(): #index1 là số thứ tự hàng của tâm cụm
            d = distance(c1, c2)
            ED.append(d)
        X["d(C"+str(i)+")"] = ED #gán khoảng cách vào data X
        #sau khi tính xong khoảng cách với tâm số 1. ta tăng i lên để tính với các tâm tiếp theo
        i = i + 1

    #Sau khi đã tính xong hết tất cả các khoảng cách. Ta sẽ gán điểm vào cụm có khoảng cách nhỏ nhất
    C = []
    for index, row in X.iterrows():
        min_kc = row["d(C1)"] #Giả sử nó gần cụm 1 nhất
        pos = 1 #Gán vị trí của nó vào cụm 1
        for i in range(0,k):
            if row["d(C"+str(i+1)+")"] < min_kc:
                min_kc =  row["d(C"+str(i+1)+")"]
                pos = i + 1
        C.append(str(pos)) #Gán vị trí của điểm vào cụm gần nhất

    X["Cum"] = C #Thêm cột gán cụm vào data X
    print("\nDữ liệu X:\n", X)

    #cập nhật lại tâm
    centroids = X.groupby(["Cum"]).mean()[['Sales', 'Quantity','Discount', 'Order ID', 'Profit']]
    print("\nTam cum cu:\n", old_centroids)
    print("\nTam cum moi:\n", centroids)

    #Tính xem còn sai khác nào không?
    # diff = (centroids - old_centroids).abs().sum().sum()
    diff = (centroids['Sales'] - old_centroids['Sales']).abs().sum() + (centroids['Quantity'] - old_centroids['Quantity']).abs().sum() + (centroids['Discount'] - old_centroids['Discount']).abs().sum() + (centroids['Order ID'] - old_centroids['Order ID']).abs().sum() + (centroids['Profit'] - old_centroids['Profit']).abs().sum()

    print("\nDiff: \n", diff)

    #Gán tâm mới kia thành tâm cũ để chuẩn bị cho quá trình lặp tiếp theo
    old_centroids = centroids.copy()
    loop += 1
    #Dùng biến j để giúp bài toán không dừng lại ở việc chỉ lặp 1 lần.
    if j == 0:
        diff = 1
        j = j + 1
    
print("\nDu lieu cuoi cung: \n", X)
print("\nTam cuoi cung:\n", centroids)
print(centroids.columns)

#Thống kê số mẫu trong cụm
cluster_count = X['Cum'].value_counts()
print("\nTong so mau trong moi cum: \n", cluster_count)

score = silhouette_score(X_scaler, X['Cum'])
print("\nMuc do phu hop Silhouette_score = ", score)

# # Tạo form
# form = Tk()  
# form.title("Dự đoán phân khúc khách hàng")
# form.geometry("1000x600")

# # Tiêu đề form
# lable_dudoan = Label(form, text="Dự đoán phân khúc khách hàng", font=("Arial", 20), fg="brown")
# lable_dudoan.grid(row=0, column=1, pady=10, sticky="e")

# # Nhóm nhập thông tin
# group1 = LabelFrame(form, text="Nhập thông tin để dự đoán", padx=10, pady=10)
# group1.grid(row=1, column=1, padx=50, pady=30)

# # Nhãn và textbox cho các trường nhập liệu
# fields = ["Sales", "Quantity", "Discount", "Order ID", "Profit"]
# textboxes = {}

# for i, field in enumerate(fields):
#     lable = Label(group1, text=f"{field}:")
#     lable.grid(row=i + 1, column=1, pady=10, sticky="e")
    
#     textbox = Entry(group1, width=30)
#     textbox.grid(row=i + 1, column=2)
#     textboxes[field] = textbox  # Lưu trữ textbox vào dictionary để sử dụng sau

# group2 = LabelFrame(form, bd=0)
# group2.grid(row=1, column=2)
# group3 = LabelFrame(group2, text="Đánh giá mô hình được chọn:")
# group3.grid(row=1, column=1, pady=20)

# lable_ketqua = Label(group2, text = "Kết quả", font=("Arial italic", 8)).grid(row = 3, column = 1, pady = 10)

# lb_num = Label(group3)
# lb_num.grid(row=1, column=1, padx = 35, pady = 20)
# centroids_str = "Cụm\tSales \t Quantity \t Discount \t Order ID \t Profit\n"
# for index, row in centroids.iterrows():
#     cluster_number = int(index)
#     centroids_str += f"{cluster_number} \t {row['Sales']:.6f} \t {row['Quantity']:.6f} \t {row['Discount']:.6f} \t {row['Order ID']:.6f} \t {row['Profit']:.6f}\n"
# lb_num.configure(text = str(centroids_str))
# lb_num = Label(group3)
# lb_num.grid(row=2, column=1, padx = 35, pady = 20)
# lb_num.configure(text = "Mức độ phù lợp Silhouette_score: "+str(score))

# def dudoan():
#         Sale = textboxes['Sales'].get()
#         quantity = textboxes['Quantity'].get()
#         discount = textboxes['Discount'].get()
#         orderid = textboxes['Order ID'].get()
#         profit = textboxes['Profit'].get()
#         if((Sale == '') or (quantity == '') or (discount == '') or (orderid == '') or (profit == '')):
#             messagebox.showinfo("Thông báo", "Bạn cần nhập đẩy đủ thông tin!")
#         else:
#             x_dudoan = pd.DataFrame({
#             'Sales': [float(Sale)],
#             'Quantity': [float(quantity)],
#             'Discount': [float(discount)],
#             'Order ID': [float(orderid)],
#             'Profit': [float(profit)]
#         })

#         min_d = float('inf')
#         C = 1
#         for index1, row in centroids.iterrows():
#             d = distance(row, x_dudoan.iloc[0])  # Truy cập hàng đầu tiên của DataFrame mới
#             if d < min_d:
#                 min_d = d
#                 C = index1
        
#         lb_pred.configure(text= "Cụm: " + str(C))

# button_1 = Button(group2, text = 'Kết quả dự đoán', font=("Arial Bold", 9), fg = "black", bg = "green", command = dudoan)
# button_1.grid(row = 2, column = 1)
# lb_pred = Label(group2, text="...", font=("Arial Bold", 9), fg = "white", bg = "SlateGray4")
# lb_pred.grid(row=4, column=1)

# # Bắt đầu vòng lặp Tkinter
# form.mainloop()


#Lay các giá trị x và y để vẽ hừn
x = X['Sales'].values
y = X['Profit'].values


colors = X['Cum'].astype(int).values  # Chuyển đổi nhãn cụm thành kiểu số

# Lấy tâm cụm
# centroids = X.groupby("Cum").mean()[['Pregnancies', "Glucose", "BloodPressure", "SkinThickness", "Insulin", "free_sulfur_dioxide", "DiabetesPedigreeFunction", "Age"]]

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=colors, marker='o', alpha=0.6, edgecolors='k', cmap='viridis')

# Vẽ tâm cụm
for i, centroid in centroids.iterrows():
    plt.scatter(centroid['Profit'], centroid['Sales'], c='red', marker='X', s=200, label=f'Centroid {i}')

# Thêm nhãn cho trục
plt.xlabel('Sales', fontsize=16)
plt.ylabel('Profit', fontsize=16)
# plt.('BloodPressure', fontsize=16)

plt.title("Customer Clustering Chart", fontsize=18)

# Thêm legend để phân biệt các tâm cụm
plt.legend(loc='upper right')

# Hiển thị biểu đồ
plt.show()


