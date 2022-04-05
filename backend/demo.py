import sys
import dex

# setup model
dex.eval()


if __name__ == '__main__':
    print('heree')
    if len(sys.argv) < 2:
        print("Usage: python demo.py path/to/img")
        sys.exit()

    path = sys.argv[1]
    age = dex.estimate(path)[0]
    print("predict image: {}".format(path))
    print("age: {:.3f}".format(age))
