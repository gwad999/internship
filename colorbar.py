import matplotlib.pyplot as plt
students = ["A", "B", "C", "D", "E"]
marks = [85, 70, 90, 65, 95]
plt.bar(students, marks, color=["red", "blue", "green", "orange", "purple"])
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.savefig("student_marks_colored.png")