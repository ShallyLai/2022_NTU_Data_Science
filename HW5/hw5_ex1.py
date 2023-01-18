#!/usr/local/bin/python3

def gini_index_calculator():
    num_c1 = float(input('Please input the number of class 1: '))
    num_c2 = float(input('Please input the number of class 2: '))
    num_total = num_c1 + num_c2
    gini_index = 1 - (num_c1/num_total) ** 2 - (num_c2/num_total) ** 2
    print('Gini index:', gini_index)
    return None

def weighted_mean():
    num_g1 = float(input('Please input the number of group 1: '))
    gini_g1 = float(input('Please input gini index of group 1: '))
    num_g2 = float(input('Please input the number of group 2: '))
    gini_g2 = float(input('Please input gini index of group 2: '))

    num_total = num_g1 + num_g2
    weight_g1 = num_g1 / num_total
    weight_g2 = num_g2 / num_total

    gini_split = gini_g1 * weight_g1 + gini_g2 * weight_g2
    print('Gini_split:', gini_split)
    return None


if __name__ == '__main__':
    while(1):
        mode = input('Welcome to Gini Index calculator.\n' +\
            'Please enter \'g\' to calculate gini index ' + \
            'or \'s\' to calculate gini_split ' +\
            'or \'q\' to quit: ')

        if(mode == 'g'):
            gini_index_calculator()
        elif(mode == 's'):
            weighted_mean()
        elif(mode == 'q'):
            print('Quit.')
            exit()
        else:
            continue
