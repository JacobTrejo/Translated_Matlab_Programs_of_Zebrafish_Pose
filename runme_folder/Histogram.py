import matplotlib
import numpy as np
import matplotlib.pyplot as plt
testingCalc_proj_w_refra_cpu = 1.5045155184375137e-29
testingBellyModel = 2.78314228e-29
testingHeadModel = 8.1454156e-30
testingEye1model = 1.181895609235414e-33
testingEye2model = 6.279142384857047e-32

total = testingCalc_proj_w_refra_cpu+testingBellyModel+testingHeadModel + testingEye1model + testingEye2model
testingCalc_proj_w_refra_cpu = testingCalc_proj_w_refra_cpu / total
testingBellyModel = testingBellyModel / total
testingHeadModel = testingHeadModel / total
testingEye1model = testingEye1model / total
testingEye2model = testingEye2model / total


data = {
    "Calc_proj_w_refra_cpu": testingCalc_proj_w_refra_cpu, "BellyModel": testingBellyModel, "HeadModel": testingHeadModel, "Eye1model": testingEye1model, 'Eye2model':testingEye2model}

programs = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (10, 5))
plt.bar(programs, values, color ='cyan',
        width = 0.4)

plt.xlabel("Python programs")
plt.ylabel("Total MS Error")
plt.title("Error of Python Programs")
plt.show()

