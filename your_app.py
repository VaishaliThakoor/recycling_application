import streamlit as st

# Define the recyclable items
recyclables = {
    'Paper': ['Newspapers', 'Magazines', 'Cardboard'],
    'Plastic': ['Bottles', 'Containers', 'Bags'],
    'Glass': ['Bottles', 'Jars'],
    'Metal': ['Cans', 'Aluminum foil'],
    'Electronics': ['Computers', 'Phones', 'Printers']
}

# Create the Streamlit app
def main():
    st.title('Recycling Guide')

    # Display instructions
    st.write('Select the type of item you want to recycle:')

    # Create a dropdown to select recyclable item
    recyclable_type = st.selectbox('Recyclable Type', list(recyclables.keys()))

    # Display the recyclable items based on the selected type
    if recyclable_type:
        st.write('You can recycle the following items:')
        for item in recyclables[recyclable_type]:
            st.write(f'- {item}')

# Run the app
if __name__ == '__main__':
    main()

