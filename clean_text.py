from constant import *
from split_audio import *

vocab = 'ẻ6ụí3ỹýẩởềõ7êứỏvỷalựqờjốàỗnéủуôuyằ4wbệễsìầỵ8dểrũcạ9ếùỡ2tiǵử̀á0ậeộmẳợĩhâúọồặfữắỳxóãổị̣zảđèừòẵ1ơkẫpấẽỉớẹăoư5|'
def clear_text(row):
  correct = [
    ['\\Delta ', 'delta '],
    ['\\arctan ', 'arctan '],
    ['\\lim ', 'lim '],
    ['\\ln ', 'ln '],
    ['\\ln(x) ', 'ln x '],
    ['\\tan ', 'tan '],
    ['\\theta ', 'theta '],
    ['\\theta) ', 'theta '],
    ['\\theta_ ', 'theta '],
    ['\\theta_0 ', 'theta không '],
    ['\\theta_1 ', 'theta một '],
    ['\\theta_1}(x) ', 'theta một x '],
    ['\\theta_2 ', 'theta hai '],
    ['\\theta_8 ', 'theta tám '],
    ['\\theta} ', 'theta '],
    ['\\theta}(j) ', 'theta j '],
    ['\\theta}(x) ', 'theta x '],
    ['\\theta}(x_1) ', 'theta x một ']
  ]

  text = row['text'].lower()
  for item in correct:
    text = text.replace(item[0], item[1])

  text = re.sub('[^' + vocab + ']', ' ', text).strip()
  text = ' '.join(text.split())

  row['text'] = text
  return row