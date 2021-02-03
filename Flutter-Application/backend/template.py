'''

ML Model steps


 1. Receive an input (image) - n fishes
 2. We segment the image - We separate all fishes possible individually
 3. 
    for fish in fishes:
        species,family,genus,WEIGHT,COST = NinjaLevelModel(fish)
4. return array

'''
def getFishesAndClasses(image):

    # Apply all your processing.
    n  = 100
    arr =  [
        {
            'species':200,
            'family':'whale',
            'genus':10,
            'weight':20,
            'cost':300,
            'count':2
        },
        {
            'species':200,
            'family':'dolphin',
            'genus':10,
            'weight':20,
            'cost':300,
            'count':2
        },
    ]
    return arr
