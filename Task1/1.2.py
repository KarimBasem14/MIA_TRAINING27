class TicketCodec:
    def __init__(self):
        self.modular_num = 37
        pass

    def weighted_sum(self, ticket):
        """
            How it works:
                For each character it multiplies its index (base 1) and its ord() and adds it to the accumlator variable (coded_sum)
        """
        coded_sum = 0
        for i, char in enumerate(ticket):
            coded_sum += (i+1) * ord(char)

        coded_sum = coded_sum % self.modular_num

        return coded_sum

    def encode(self, ticket_id):
        coded_sum = self.weighted_sum(ticket_id)

        return ticket_id + "-" + str(coded_sum)

    def decode(self, barcode):
        # Extract coded sum for ticket
        split_barcode = barcode.split("-")
        ticket = split_barcode[0]
        sent_sum = split_barcode[1]


        # Rerun sum algorithm
        computed_sum = self.weighted_sum(ticket)
        
        # Check if computed value equals the encoded value
        if str(computed_sum) == sent_sum:
            print("Correct ticket, returned ticket")
            return ticket
        else:
            print("False ticket")


if __name__ == "__main__":
    codec = TicketCodec()

    test_ids = [
        "MIA2026Gate7",
        "EPICTASK",
        "SoWhereWeGoAsWeWalkThisLonelyRoad",
        "Andalus",
        "KimoCono14",
    ]

    
    encoded_tickets = [codec.encode(tid) for tid in test_ids]

    print("--- Testing Original Valid Barcodes ---")
    for barcode in encoded_tickets:
        codec.decode(barcode)

    # Now peform error on some of them

    # All corrupted except middle one
    corrupted_test_ids = [
        "MIA2026Gate4",
        "EPIcTASK",
        "SoWhereWeGoAsWeWalkThisLonelyRoad",
        "Andalusyia",
        "KimoCono19",
    ]

    for i in range(len(corrupted_test_ids)):
        corrupted_test_ids[i] += "-" + str(codec.weighted_sum(test_ids[i]))

    # corrupted_data = [corrupted_test_ids[i] + "-" + str(codec.weighted_sum(test_ids[i])) for i in range(len(corrupted_test_ids))]

    print("--- Testing Corrupted Data ---")
    for barcode in corrupted_test_ids:
        codec.decode(barcode)




