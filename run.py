from backend.app import app
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)