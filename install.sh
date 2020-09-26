GREEN="\e[92m"
RED="\e[91m"
NORMAL="\e[0m"
SUCCSESS="$GREEN Success! $NORMAL"
FAIL="$RED Failed ... $NORMAL"

echo -n "Creating python virtual environment: "
python3 -m venv env && echo -e $SUCCSESS || echo -e $FAIL

echo -n "Activating virtual environment: "
source env/bin/activate && echo -e $SUCCSESS || echo -e $FAIL

echo -n "Installing wheel: "
pip3 install wheel && echo -e $SUCCSESS || echo -e $FAIL

echo -n "Pip install requirements.txt: "
pip3 install -r requirements.txt > /dev/null && echo -e $SUCCSESS || echo -e $FAIL
