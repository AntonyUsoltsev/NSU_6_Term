create table cashier (
    cashier_id bigserial not null,
    name varchar(255) not null,
    second_name varchar(255) not null,
    primary key (cashier_id)
);

create table customer (
    customer_id bigserial not null,
    email varchar(255) not null,
    name varchar(255) not null,
    second_name varchar(255) not null,
    primary key (customer_id)
);

create table delivery (
    delivery_id bigserial not null,
    delivery_date timestamp(6) not null,
    supplier_id bigint not null,
    primary key (delivery_id)
);

create table delivery_list (
    item_id bigint not null,
    delivery_id bigint not null,
    amount bigint not null,
    primary key (item_id, delivery_id)
);

create table items (
    item_id bigserial not null,
    amount integer not null,
    category varchar(255) not null,
    cell_number integer not null,
    defect_amount integer not null,
    name varchar(255) not null,
    price integer not null,
    primary key (item_id)
);

create table order_list (
    order_id bigint not null,
    item_id bigint not null,
    amount bigint not null,
    primary key (order_id, item_id)
);

create table orders (
    order_id bigserial not null,
    full_price integer not null,
    order_date timestamp(6) not null,
    customer_id bigint not null,
    primary key (order_id)
);

create table supplier (
    supplier_id bigserial not null,
    documents varchar(255) not null,
    garanty boolean not null,
    name varchar(255) not null,
    type varchar(255) not null,
    primary key (supplier_id)
);

create table transaction (
    transaction_id bigserial not null,
    transaction_date timestamp(6) not null,
    transaction_type varchar(255) not null,
    cashier_id bigint not null,
    order_id bigint not null unique,
    primary key (transaction_id)
);


alter table if exists customer 
       add constraint email_unique unique (email);
       
alter table if exists delivery 
       add constraint fk_sp_id foreign key (supplier_id) references supplier;

alter table if exists delivery_list 
       add constraint fk_del_id foreign key (delivery_id) references delivery;
       
alter table if exists delivery_list 
       add constraint fk_item_id foreign key (item_id) references items;

alter table if exists order_list 
       add constraint fk_item2_id foreign key (item_id) references items;
 
alter table if exists order_list 
       add constraint fk_ord_id foreign key (order_id) references orders;
       
alter table if exists orders 
       add constraint fk_cus_id foreign key (customer_id) references customer;

alter table if exists transaction
       add constraint fk_cash_id foreign key (cashier_id) references cashier;
       
alter table if exists transaction
       add constraint fk_ord2_id foreign key (order_id) references orders;