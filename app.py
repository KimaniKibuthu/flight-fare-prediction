import joblib
import streamlit as st

@st.cache(allow_output_mutation=True, show_spinner=False)
def model_loader():
    with open("forest.pkl", "rb") as input_model:
        classifier = joblib.load(input_model)
    return classifier


def time_extractor(time):
    hour = time.hour
    minute = time.minute
    return hour, minute


def encoder(encoding, to_encode):
    val = None
    for value, name in encoding:
        if to_encode == name:
            val = value

    if to_encode not in encoding:
        val = -1

    return val


def feature_creator(departure_date, arrival_date, departure_time, arrival_time, airline, destination,
                    origin, route, total_stops, additional_info):
    # Define features

    arrival_hour, arrival_minute = time_extractor(arrival_time)
    departure_hour, departure_minute = time_extractor(departure_time)

    total_duration = ((arrival_hour * 60) + arrival_minute) - ((departure_hour * 60) + departure_minute)
    day = departure_date.day
    month = departure_date.month

    stops_airline = total_stops + '_' + airline

    # Convert to numerical values
    # Encodings
    encoder_data = [(3, '0_IndiGo'), (18, '2_Air India'), (20, '2_Jet Airways'), (11, '1_IndiGo'), (6, '0_SpiceJet'),
                    (12, '1_Jet Airways'), (13, '1_Multiple carriers'), (9, '1_Air India'), (1, '0_Air India'),
                    (15, '1_SpiceJet'), (10, '1_GoAir'), (7, '0_Vistara'), (0, '0_Air Asia'), (4, '0_Jet Airways'),
                    (16, '1_Vistara'), (2, '0_GoAir'), (21, '2_Multiple carriers'), (19, '2_IndiGo'), (8, '1_Air Asia'),
                    (23, '3_Air India'), (5, '0_Other'), (14, '1_Other'), (17, '2_Air Asia'),
                    (24, '3_Multiple carriers'), (22, '2_Other'), (25, '4_Air India')]

    airline_encoding = [(3, 'IndiGo'), (1, 'Air India'), (4, 'Jet Airways'), (7, 'SpiceJet'), (5, 'Multiple carriers'),
                        (2, 'GoAir'), (8, 'Vistara'), (0, 'Air Asia'), (6, 'Other')]

    information_encoding = [(3, 'No info'), (6, 'other'), (2, 'No check-in baggage included'), (5, 'layover'),
                            (1, 'Change airports'), (0, 'Business class'), (4, 'Other')]

    route_encoding = [(18, 'BLR → DEL'), (84, 'CCU → IXR → BBI → BLR'), (118, 'DEL → LKO → BOM → COK'),
                      (91, 'CCU → NAG → BLR'), (29, 'BLR → NAG → DEL'), (64, 'CCU → BLR'), (5, 'BLR → BOM → DEL'),
                      (104, 'DEL → BOM → COK'), (103, 'DEL → BLR → COK'), (127, 'MAA → CCU'), (66, 'CCU → BOM → BLR'),
                      (97, 'DEL → AMD → BOM → COK'), (123, 'DEL → PNQ → COK'), (105, 'DEL → CCU → BOM → COK'),
                      (17, 'BLR → COK → DEL'), (113, 'DEL → IDR → BOM → COK'), (119, 'DEL → LKO → COK'),
                      (78, 'CCU → GAU → DEL → BLR'), (122, 'DEL → NAG → BOM → COK'), (90, 'CCU → MAA → BLR'),
                      (111, 'DEL → HYD → COK'), (80, 'CCU → HYD → BLR'), (106, 'DEL → COK'), (73, 'CCU → DEL → BLR'),
                      (3, 'BLR → BOM → AMD → DEL'), (45, 'BOM → DEL → HYD'), (121, 'DEL → MAA → COK'),
                      (48, 'BOM → HYD'), (102, 'DEL → BHO → BOM → COK'), (116, 'DEL → JAI → BOM → COK'),
                      (99, 'DEL → ATQ → BOM → COK'), (117, 'DEL → JDH → BOM → COK'), (61, 'CCU → BBI → BOM → BLR'),
                      (28, 'BLR → MAA → DEL'), (108, 'DEL → GOI → BOM → COK'), (101, 'DEL → BDQ → BOM → COK'),
                      (87, 'CCU → JAI → BOM → BLR'), (60, 'CCU → BBI → BLR'), (24, 'BLR → HYD → DEL'),
                      (125, 'DEL → TRV → COK'), (85, 'CCU → IXR → DEL → BLR'), (115, 'DEL → IXU → BOM → COK'),
                      (82, 'CCU → IXB → BLR'), (9, 'BLR → BOM → JDH → DEL'), (126, 'DEL → UDR → BOM → COK'),
                      (112, 'DEL → HYD → MAA → COK'), (67, 'CCU → BOM → COK → BLR'), (15, 'BLR → CCU → DEL'),
                      (68, 'CCU → BOM → GOI → BLR'), (124, 'DEL → RPR → NAG → BOM → COK'),
                      (110, 'DEL → HYD → BOM → COK'), (72, 'CCU → DEL → AMD → BLR'), (93, 'CCU → PNQ → BLR'),
                      (16, 'BLR → CCU → GAU → DEL'), (74, 'CCU → DEL → COK → BLR'), (30, 'BLR → PNQ → DEL'),
                      (51, 'BOM → JDH → DEL → HYD'), (4, 'BLR → BOM → BHO → DEL'), (98, 'DEL → AMD → COK'),
                      (27, 'BLR → LKO → DEL'), (77, 'CCU → GAU → BLR'), (46, 'BOM → GOI → HYD'),
                      (65, 'CCU → BOM → AMD → BLR'), (63, 'CCU → BBI → IXR → DEL → BLR'),
                      (107, 'DEL → DED → BOM → COK'), (120, 'DEL → MAA → BOM → COK'), (0, 'BLR → AMD → DEL'),
                      (33, 'BLR → VGA → DEL'), (88, 'CCU → JAI → DEL → BLR'), (59, 'CCU → AMD → BLR'),
                      (95, 'CCU → VNS → DEL → BLR'), (6, 'BLR → BOM → IDR → DEL'), (1, 'BLR → BBI → DEL'),
                      (20, 'BLR → GOI → DEL'), (36, 'BOM → AMD → ISK → HYD'), (44, 'BOM → DED → DEL → HYD'),
                      (114, 'DEL → IXC → BOM → COK'), (92, 'CCU → PAT → BLR'), (12, 'BLR → CCU → BBI → DEL'),
                      (62, 'CCU → BBI → HYD → BLR'), (10, 'BLR → BOM → NAG → DEL'), (13, 'BLR → CCU → BBI → HYD → DEL'),
                      (19, 'BLR → GAU → DEL'), (39, 'BOM → BHO → DEL → HYD'), (53, 'BOM → JLR → HYD'),
                      (25, 'BLR → HYD → VGA → DEL'), (89, 'CCU → KNU → BLR'), (70, 'CCU → BOM → PNQ → BLR'),
                      (100, 'DEL → BBI → COK'), (34, 'BLR → VGA → HYD → DEL'), (52, 'BOM → JDH → JAI → DEL → HYD'),
                      (109, 'DEL → GWL → IDR → BOM → COK'), (94, 'CCU → RPR → HYD → BLR'), (96, 'CCU → VTZ → BLR'),
                      (76, 'CCU → DEL → VGA → BLR'), (7, 'BLR → BOM → IDR → GWL → DEL'),
                      (75, 'CCU → DEL → COK → TRV → BLR'), (43, 'BOM → COK → MAA → HYD'), (55, 'BOM → NDC → HYD'),
                      (2, 'BLR → BDQ → DEL'), (71, 'CCU → BOM → TRV → BLR'), (69, 'CCU → BOM → HBX → BLR'),
                      (38, 'BOM → BDQ → DEL → HYD'), (42, 'BOM → CCU → HYD'), (32, 'BLR → TRV → COK → DEL'),
                      (26, 'BLR → IDR → DEL'), (86, 'CCU → IXZ → MAA → BLR'), (79, 'CCU → GAU → IMF → DEL → BLR'),
                      (47, 'BOM → GOI → PNQ → HYD'), (40, 'BOM → BLR → CCU → BBI → HYD'), (54, 'BOM → MAA → HYD'),
                      (11, 'BLR → BOM → UDR → DEL'), (57, 'BOM → UDR → DEL → HYD'), (35, 'BLR → VGA → VTZ → DEL'),
                      (22, 'BLR → HBX → BOM → BHO → DEL'), (81, 'CCU → IXA → BLR'), (56, 'BOM → RPR → VTZ → HYD'),
                      (21, 'BLR → HBX → BOM → AMD → DEL'), (49, 'BOM → IDR → DEL → HYD'), (41, 'BOM → BLR → HYD'),
                      (31, 'BLR → STV → DEL'), (83, 'CCU → IXB → DEL → BLR'), (50, 'BOM → JAI → DEL → HYD'),
                      (58, 'BOM → VNS → DEL → HYD'), (23, 'BLR → HBX → BOM → NAG → DEL'), (8, 'BLR → BOM → IXC → DEL'),
                      (14, 'BLR → CCU → BBI → HYD → VGA → DEL'), (37, 'BOM → BBI → HYD')]

    origin_encoding = [(0, 'Banglore'), (3, 'Kolkata'), (2, 'Delhi'), (1, 'Chennai'), (4, 'Mumbai')]

    destination_encoding = [(5, 'New Delhi'), (0, 'Banglore'), (1, 'Cochin'), (4, 'Kolkata'), (2, 'Delhi'),
                            (3, 'Hyderabad')]

    # Encoded features
    new_stops_airline = encoder(encoder_data, stops_airline)
    new_airline = encoder(airline_encoding, airline)
    new_information = encoder(information_encoding, additional_info)
    new_route = encoder(route_encoding, route)
    new_source = encoder(origin_encoding, origin)
    new_destination = encoder(destination_encoding, destination)

    # Define final features
    list_of_features = [new_stops_airline, day, total_duration, month, new_airline,
                        new_information, new_route, departure_hour, arrival_hour, new_source,
                        new_destination, departure_minute, arrival_minute, total_stops]

    return list_of_features


def predict_delay(departure_date, arrival_date, departure_time, arrival_time, airline, destination,
                  origin, route, total_stops, additional_info):
    """Flight Delay Predictions
    This is using docstrings for specifications.
    ---
    parameters:
      - name: time
        in: query
        type: datetime
        required: true
      - name: carrier
        in: query
        type: str
        required: true
      - name: destination
        in: query
        type: str
        required: true
      - name: origin
        in: query
        type: str
        required: true
      - name: month
        in: query
        type: number
        required: true

    responses:
        200:
            description: The output values

    """
    classifier = model_loader()
    to_predict = feature_creator(departure_date, arrival_date, departure_time, arrival_time, airline, destination,
                                 origin, route, total_stops, additional_info)
    prediction = int(classifier.predict([to_predict]))
    return prediction


def main():
    st.title("Flight Fare Prediction")
    html_temp = """
    <div style="background-color:RebeccaPurple;padding:10px">
    <h2 style="color:white;text-align:center;">Flight Delay Prediction Web App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    departure_date = st.date_input('Date of departure')

    arrival_date = st.date_input('Date of arrival')

    departure_time = st.time_input('Time of Departure')

    arrival_time = st.time_input('Time of Arrival')

    airline = st.selectbox('Flying with', ('IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
                                           'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia', 'Other'))

    destination = st.selectbox('Destination Airport', ('New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi',
                                                       'Hyderabad'))

    origin = st.selectbox('Origin Airport', ('Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'),
                          key='origin')

    route = st.selectbox('Route', ('BLR → DEL', 'CCU → IXR → BBI → BLR', 'DEL → LKO → BOM → COK',
                                   'CCU → NAG → BLR', 'BLR → NAG → DEL', 'CCU → BLR',
                                   'BLR → BOM → DEL', 'DEL → BOM → COK', 'DEL → BLR → COK',
                                   'MAA → CCU', 'CCU → BOM → BLR', 'DEL → AMD → BOM → COK',
                                   'DEL → PNQ → COK', 'DEL → CCU → BOM → COK', 'BLR → COK → DEL',
                                   'DEL → IDR → BOM → COK', 'DEL → LKO → COK',
                                   'CCU → GAU → DEL → BLR', 'DEL → NAG → BOM → COK',
                                   'CCU → MAA → BLR', 'DEL → HYD → COK', 'CCU → HYD → BLR',
                                   'DEL → COK', 'CCU → DEL → BLR', 'BLR → BOM → AMD → DEL',
                                   'BOM → DEL → HYD', 'DEL → MAA → COK', 'BOM → HYD',
                                   'DEL → BHO → BOM → COK', 'DEL → JAI → BOM → COK',
                                   'DEL → ATQ → BOM → COK', 'DEL → JDH → BOM → COK',
                                   'CCU → BBI → BOM → BLR', 'BLR → MAA → DEL',
                                   'DEL → GOI → BOM → COK', 'DEL → BDQ → BOM → COK',
                                   'CCU → JAI → BOM → BLR', 'CCU → BBI → BLR', 'BLR → HYD → DEL',
                                   'DEL → TRV → COK', 'CCU → IXR → DEL → BLR',
                                   'DEL → IXU → BOM → COK', 'CCU → IXB → BLR',
                                   'BLR → BOM → JDH → DEL', 'DEL → UDR → BOM → COK',
                                   'DEL → HYD → MAA → COK', 'CCU → BOM → COK → BLR',
                                   'BLR → CCU → DEL', 'CCU → BOM → GOI → BLR',
                                   'DEL → RPR → NAG → BOM → COK', 'DEL → HYD → BOM → COK',
                                   'CCU → DEL → AMD → BLR', 'CCU → PNQ → BLR',
                                   'BLR → CCU → GAU → DEL', 'CCU → DEL → COK → BLR',
                                   'BLR → PNQ → DEL', 'BOM → JDH → DEL → HYD',
                                   'BLR → BOM → BHO → DEL', 'DEL → AMD → COK', 'BLR → LKO → DEL',
                                   'CCU → GAU → BLR', 'BOM → GOI → HYD', 'CCU → BOM → AMD → BLR',
                                   'CCU → BBI → IXR → DEL → BLR', 'DEL → DED → BOM → COK',
                                   'DEL → MAA → BOM → COK', 'BLR → AMD → DEL', 'BLR → VGA → DEL',
                                   'CCU → JAI → DEL → BLR', 'CCU → AMD → BLR',
                                   'CCU → VNS → DEL → BLR', 'BLR → BOM → IDR → DEL',
                                   'BLR → BBI → DEL', 'BLR → GOI → DEL', 'BOM → AMD → ISK → HYD',
                                   'BOM → DED → DEL → HYD', 'DEL → IXC → BOM → COK',
                                   'CCU → PAT → BLR', 'BLR → CCU → BBI → DEL',
                                   'CCU → BBI → HYD → BLR', 'BLR → BOM → NAG → DEL',
                                   'BLR → CCU → BBI → HYD → DEL', 'BLR → GAU → DEL',
                                   'BOM → BHO → DEL → HYD', 'BOM → JLR → HYD',
                                   'BLR → HYD → VGA → DEL', 'CCU → KNU → BLR',
                                   'CCU → BOM → PNQ → BLR', 'DEL → BBI → COK',
                                   'BLR → VGA → HYD → DEL', 'BOM → JDH → JAI → DEL → HYD',
                                   'DEL → GWL → IDR → BOM → COK', 'CCU → RPR → HYD → BLR',
                                   'CCU → VTZ → BLR', 'CCU → DEL → VGA → BLR',
                                   'BLR → BOM → IDR → GWL → DEL', 'CCU → DEL → COK → TRV → BLR',
                                   'BOM → COK → MAA → HYD', 'BOM → NDC → HYD', 'BLR → BDQ → DEL',
                                   'CCU → BOM → TRV → BLR', 'CCU → BOM → HBX → BLR',
                                   'BOM → BDQ → DEL → HYD', 'BOM → CCU → HYD',
                                   'BLR → TRV → COK → DEL', 'BLR → IDR → DEL',
                                   'CCU → IXZ → MAA → BLR', 'CCU → GAU → IMF → DEL → BLR',
                                   'BOM → GOI → PNQ → HYD', 'BOM → BLR → CCU → BBI → HYD',
                                   'BOM → MAA → HYD', 'BLR → BOM → UDR → DEL',
                                   'BOM → UDR → DEL → HYD', 'BLR → VGA → VTZ → DEL',
                                   'BLR → HBX → BOM → BHO → DEL', 'CCU → IXA → BLR',
                                   'BOM → RPR → VTZ → HYD', 'BLR → HBX → BOM → AMD → DEL',
                                   'BOM → IDR → DEL → HYD', 'BOM → BLR → HYD', 'BLR → STV → DEL',
                                   'CCU → IXB → DEL → BLR', 'BOM → JAI → DEL → HYD',
                                   'BOM → VNS → DEL → HYD', 'BLR → HBX → BOM → NAG → DEL',
                                   'BLR → BOM → IXC → DEL', 'BLR → CCU → BBI → HYD → VGA → DEL',
                                   'BOM → BBI → HYD'))

    total_stops = st.selectbox('Total stops ',
                               ('0', '1', '2', '3', '4'))

    additional_info = st.selectbox('Any additional information',
                                   ['No info', 'other', 'No check-in baggage included', 'layover',
                                    'Change airports', 'Business class', 'Other'])

    if st.button("Predict"):
        result = predict_delay(departure_date, arrival_date, departure_time, arrival_time, airline, destination,
                               origin, route, total_stops, additional_info)
        st.success('The price of the flight is Rs. {}.'.format(result))


if __name__ == '__main__':
    main()
