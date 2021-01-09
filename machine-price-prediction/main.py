import uvicorn
from fastapi import FastAPI
from predict import predict_price


app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Velkommen!'}


@app.get('/{ID}')
def return_predict_price(ID: int):
    print(f'Evaluating ID {ID}...')
    prediction = round(predict_price(ID=ID))
    print(f'Predicted price: {prediction}')
    return {
        'prediction': prediction,
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)