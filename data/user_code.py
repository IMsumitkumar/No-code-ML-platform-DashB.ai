from django.contrib import messages
from yapf.yapflib.yapf_api import FormatCode

def user_given(request, data, col, code):

    try:
        x = FormatCode(code)
        exec(x[0])
    except Exception as e:
        messages.error(request, "Invalid Argument parsing or go to previous page and rerun")

