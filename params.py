
mi_path = 'supermarket/Mi_6'
sam_path = 'supermarket/Samsung_C5Pro'
MNIST_ROOT = './mnist/'

EPOCH = 50
BATCH_SIZE = 1
HIDDEN_SIZE = 64
TIME_STEP = 10
INPUT_SIZE = 3
OUTPUT_SIZE = 2
LR = 0.1
CLA_COUNT = 100


def show_img(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


'''
DOWNLOAD_MNIST = True
MNIST_ROOT = './mnist/'
if not(os.path.exists(MNIST_ROOT)) or not os.listdir(MNIST_ROOT):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root=MNIST_ROOT,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(root=MNIST_ROOT, train=False)

train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

TEST_DATA_SIZE = 2000
test_x = torch.unsqueeze(test_data.test_data[:TEST_DATA_SIZE], dim=1).type(torch.FloatTensor)/255
test_y = test_data.test_labels[:TEST_DATA_SIZE]

# steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # 0 ~ 2*pi (6.28)
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# show the cos & sin
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.legend(loc='best')
# plt.show()
# print(steps)



        # plotting
        # ylim_value = 1.2
        # x_steps = steps + np.ones(TIME_STEP) * TIME_STEP * (step % CLA_COUNT)
        # clear_flag = False
        # if step % CLA_COUNT == 0:
        #     clear_flag = True
        #     print("hehe")
        # if clear_flag:
        #     plt.cla()
'''

