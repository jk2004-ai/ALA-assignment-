class UpiId:
     def __init__(self, id, bank_id):
         self.my_id = id
         self.my_bank_id = bank_id
     def __repr__(self):
         return "upi" + self.my_id + "@"+ self.my_bank_id
     def __eq__(self, other):
         return self.my_id == other.my_id and self.my_bank_id == other.my_bank_id
         

jk_upi_id = UpiId("9731077575","okhdfc")
print(jk_upi_id)

