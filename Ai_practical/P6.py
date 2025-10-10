"""
#501(prac6)---01/08/19---
#AIM: Implement Naive-Bayes learning algo for RWP(Restaurant Waiting Problem).
"""

from RWP import rwp_examples
total_exp = 12
def tot(attribute, value):
    count = 0
    for key, val in rwp_examples.items():
        for key1, val1 in val.items():
            if key1 == attribute:
                if val1 == value:
                    count += 1
    return count
def getProbab(attribute, attribval, value):
    count = 0
    for key, val in rwp_examples.items():
        val1 = rwp_examples[key][attribute]
        val2 = rwp_examples[key]['ans']
        if val1 == attribval and val2 == value:
            count += 1
    probab = count / tot('ans', value)
    return probab
def main():
    PAltYes = tot('Alt', 'Y') / total_exp
    PAltNo = tot('Alt', 'N') / total_exp    
    PBarYes = tot('Bar', 'Y') / total_exp
    PBarNo = tot('Bar', 'N') / total_exp    
    PFriYes = tot('Fri', 'Y') / total_exp
    PFriNo = tot('Fri', 'N') / total_exp    
    PHunYes = tot('Hun', 'Y') / total_exp
    PHunNo = tot('Hun', 'N') / total_exp    
    PPatSome = tot('Pat', 'S') / total_exp
    PPatFull = tot('Pat', 'F') / total_exp
    PPatNone = tot('Pat', 'N') / total_exp
    PPriceCheap = tot('Price', '$') / total_exp
    PPriceAvg = tot('Price', '$$') / total_exp
    PPriceExp = tot('Price', '$$$') / total_exp    
    PRainYes = tot('Rain', 'Y') / total_exp
    PRainNo = tot('Rain', 'N') / total_exp    
    PResYes = tot('Res', 'Y') / total_exp
    PResNo = tot('Res', 'N') / total_exp    
    PTypeFrench = tot('Type', 'F') / total_exp
    PTypeItalian = tot('Type', 'I') / total_exp
    PTypeBurger = tot('Type', 'B') / total_exp
    PTypeThai = tot('Type', 'T') / total_exp    
    PEstFew = tot('Est', '0-10') / total_exp
    PEstMore = tot('Est', '10-30') / total_exp
    PEstStillMore = tot('Est', '30-60') / total_exp
    PEstTooMuch = tot('Est', '>60') / total_exp
    PAnsYes = tot('ans', 'Y') / total_exp
    PAnsNo = tot('ans', 'N') / total_exp    
    print('Probability for will wait if there is an Alternate Restaurant Nearby: ')
    print('Yes: Will Wait ', (getProbab('Alt', 'Y', 'Y') * PAnsYes/PAltYes) * 100, '%')
    print('No: Will Wait ', (getProbab('Alt', 'Y', 'N') * PAnsNo/PAltYes ) * 100, '%')
    print('Probability for will wait if there No is an Alternate Restaurant Nearby: ')
    print('Yes: Will Wait ', (getProbab('Alt', 'N', 'Y') * PAnsYes/PAltNo) * 100, '%')
    print('No: Will Wait ', (getProbab('Alt', 'N', 'N') * PAnsNo/PAltNo) * 100, '%')
    print('Probability for will wait if Estimated Wait time is 0-10 minutes: ')
    print('Yes: Will Wait ', (getProbab('Est', '0-10', 'Y') * PAnsYes/PEstFew) * 100, '%')
    print('No: Will Wait ', (getProbab('Est', '0-10', 'N') * PAnsNo/PEstFew) * 100, '%')
    print('Probability for will wait if Estimated Wait time is 10-30 minutes ')
    print('Yes: Will Wait ', (getProbab('Est', '10-30', 'Y') * PAnsYes/PEstMore) * 100, '%')
    print('No: Will Wait ', (getProbab('Est', '10-30', 'N') * PAnsNo/PEstMore) * 100, '%')
    print("Probability for Will Wait if the Estimated Wait Time is 30-60 mins: ")
    print("Yes: Will Wait: ",(getProbab('Est','30-60','Y')*PAnsYes/PEstStillMore)*100,"%")
    print("No: Will Wait: ",(getProbab('Est','30-60','N')*PAnsNo/PEstStillMore)*100,"%")
    print("Probability for Will Wait if the Estimated Wait Time is >60 mins: ")
    print("Yes: Will Wait: ",(getProbab('Est','>60','Y')*PAnsYes/PEstTooMuch)*100,"%")
    print("No: Will Wait: ",(getProbab('Est','>60','N')*PAnsNo/PEstTooMuch)*100,"%")
    print('Probability for will wait if there are Some Patrons ')
    print('Yes: Will Wait ', (getProbab('Pat', 'S', 'Y') * PAnsYes/PPatSome) * 100, '%')
    print('No: Will Wait ', (getProbab('Pat', 'S', 'N') * PAnsNo/PPatSome) * 100, '%')
    print("Probability for Will Wait if there are None Patrons: ")
    print("Yes: Will Wait: ",(getProbab('Pat','N','Y')*PAnsYes/PPatNone)*100,"%")
    print("No: Will Wait: ",(getProbab('Pat','N','N')*PAnsNo/PPatNone)*100,"%")
    print("Probability for Will Wait if there are Full Patrons: ")
    print("Yes: Will Wait: ",(getProbab('Pat','F','Y')*PAnsYes/PPatFull)*100,"%")
    print("No: Will Wait: ",(getProbab('Pat','F','N')*PAnsNo/PPatFull)*100,"%")
    print('Probability for will wait if the place is Thai ')
    print('Yes: Will Wait ', (getProbab('Type', 'T', 'Y') * PAnsYes/PTypeThai) * 100, '%')
    print('No: Will Wait ', (getProbab('Type', 'T', 'N') * PAnsNo/PTypeThai) * 100, '%')    
main()

"""
OUTPUT:

Probability for will wait if there is an Alternate Restaurant Nearby: 
Yes: Will Wait  50.0 %
No: Will Wait  50.0 %
Probability for will wait if there No is an Alternate Restaurant Nearby: 
Yes: Will Wait  50.0 %
No: Will Wait  50.0 %
Probability for will wait if Estimated Wait time is 0-10 minutes: 
Yes: Will Wait  71.42857142857143 %
No: Will Wait  28.57142857142857 %
Probability for will wait if Estimated Wait time is 10-30 minutes 
Yes: Will Wait  50.0 %
No: Will Wait  50.0 %
Probability for Will Wait if the Estimated Wait Time is 30-60 mins: 
Yes: Will Wait:  0.0 %
No: Will Wait:  100.0 %
Probability for Will Wait if the Estimated Wait Time is >60 mins: 
Yes: Will Wait:  0.0 %
No: Will Wait:  100.0 %
Probability for will wait if there are Some Patrons 
Yes: Will Wait  100.0 %
No: Will Wait  0.0 %
Probability for Will Wait if there are None Patrons: 
Yes: Will Wait:  0.0 %
No: Will Wait:  100.0 %
Probability for Will Wait if there are Full Patrons: 
Yes: Will Wait:  33.33333333333333 %
No: Will Wait:  66.66666666666666 %
Probability for will wait if the place is Thai 
Yes: Will Wait  50.0 %
No: Will Wait  50.0 %
"""
