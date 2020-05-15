import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])
s1 = torch.sparse.FloatTensor(i, v, torch.Size([2, 3])).to(device)

print(s1)
print(s1.to_dense())
print()

i = torch.LongTensor([[0, 0, 0], [0, 1, 2]])
v = torch.FloatTensor([3, 4, 5])
s2 = torch.sparse.FloatTensor(i, v, torch.Size([2, 3])).to(device)

print(s2)
print(s2.to_dense())
print()


print(s1 + s2)
print((s1 + s2).coalesce())
print((s1 + s2).to_dense())

