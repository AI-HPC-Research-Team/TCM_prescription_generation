import sys
from torch_geometric.nn import Node2Vec
import torch
import pandas as pd
from torch_geometric.data import Data as gData



def main():
    def data_preparation():
        # Edge has the format of {source:node_id, target:node_id, weight(optional)}
        edge = pd.read_csv('Edge.csv')
        # Node has the format of {node_id, node_name, label(optional)}
        node = pd.read_csv('Nodes.csv', encoding='gbk')
        node_num = len(node['Nodes'].tolist())
        source = edge['source'].tolist()
        target = edge['target'].tolist()
        edge_index = torch.LongTensor([source, target])
        train_mask = [False]*node_num
        test_mask = [False]*node_num
        for i in range(int(0.8*node_num)):
            train_mask[i] = True
        for i in range(int(0.8*node_num),node_num):
            test_mask[i] = True
        train_set = torch.tensor(train_mask,dtype=torch.bool)
        test_set = torch.tensor(test_mask,dtype=torch.bool)
        x = torch.zeros((node_num, #embedding_dim#))
        data = gData(x=x, edge_index=edge_index, y=y, train_mask=train_set, test_mask=test_set)
        return data

    def node2vec():
        data = data_preparation()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Hyperparameter can be modified according to the graph structure
        model = Node2Vec(data.edge_index, embedding_dim=#embedding_dim#, walk_length=#walk_length#,
                         context_size=#context_size#, walks_per_node=#walks_per_node#,
                         num_negative_samples=1, p=1, q=1, sparse=True).to(device)
        num_workers = 0 if sys.platform.startswith('win') else 8
        loader = model.loader(batch_size=#batch_size#, shuffle=True,
                              num_workers=num_workers)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=#learning_rate#)

        def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        @torch.no_grad()
        def test():
            model.eval()
            z = model()
            acc = model.test(z[data.train_mask], data.y[data.train_mask],
                             z[data.test_mask], data.y[data.test_mask],
                             max_iter=150)
            return acc

        def graph_embedding():
            model.eval()
            out = model(torch.arange(data.num_nodes, device=device))
            return out

        max_acc = 0
        for epoch in range(1, 5000):
            log = open('logs.txt', mode='a')
            loss = train()
            acc = test()
            if acc > max_acc:
                max_acc = acc
                out = graph_embedding()
                torch.save(out, "./node_embedding.pt")
                torch.save(model.state_dict(), "./node2vec_parameter.pkl")
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
            log.write(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}\n')
            log.close()
        return out

    graph_embedding = node2vec()


if __name__ == "__main__":
    main()
