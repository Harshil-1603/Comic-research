def collate(batch):
    imgs, texts, ys = zip(*batch)
    return list(imgs), list(texts), ys
