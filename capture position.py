import arrow

#gets the license plate number
def get_license_plate_number():
    print("Enter license plate number")
    return input()

#reports and stores the entry time
def get_entry_time():
    # return arrow.now()
    # use dummytime for now for better demonstration
    s = '2020-08-01 23:30:45'
    entry = arrow.get(s, 'YYYY-MM-DD HH:mm:ss')
    return entry
#calculates parking fees
def parking_fee(entry):
    exit = arrow.now()
    duration = exit - entry
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    total = hours * 60 + minutes
    fee = total * 3
    print("Total duration = ", hours, " : ", minutes)
    print("Parking Fee : ", fee)
    return fee

#checks if the given car is in the lot, outputs time and fees
def capture_position(platenum):
    flag =0
    for key in spots:
        if spots[key].lpn is platenum:
            print("entry time : ", parking_fee(spots[key].entry))
            print("parking spot : ", key)
            flag =1
            break
    if flag==0:
        print("could not find the car")
    return

# car object with attributes of vehicles in parking lot
class car:
    def __init__(self, lpn, entry):
        self.entry=entry
        self.lpn=lpn
tot_spots=2 # total number of spots, input from task 1
spots={} # dictionry format license plate : spot
spot_ID = ["1A", "2B", "3C", "4D", "5E"] # spot numbering - as required, dummy list for 5 spots
cars = list() # append when new car enters
for i in range(tot_spots): # loop fpr demo
    cars.append(car(get_license_plate_number(), get_entry_time()))
    spots[spot_ID[i]] = cars[i]
#function to be called when a new car enters
def entry():
    print("Welcome to smart parking system")
    c=car(get_license_plate_number(), get_entry_time())
    cars.append()
    print("license plate number : ", c.lpn)
    print("entry time : ", c.entry)
#function to be called when a car exits
def exit():
    print(" Enter license plate number of the exiting car ")
    capture_position(input())
exit()





