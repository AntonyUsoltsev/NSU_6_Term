create table if not exists supplier_type
(
    type_id   bigserial    not null,
    type_name varchar(255) not null,
    primary key (type_id)
);

create table if not exists supplier
(
    supplier_id bigserial    not null,
    documents   varchar(255),
    garanty     boolean      not null,
    name        varchar(255) not null,
    type_id     bigint       not null,
    primary key (supplier_id),
    foreign key (type_id) references supplier_type (type_id)
);

create table if not exists delivery
(
    delivery_id   bigserial    not null,
    delivery_date timestamp(6) not null,
    supplier_id   bigint       not null,
    primary key (delivery_id),
    foreign key (supplier_id) references supplier (supplier_id)
);

create table if not exists item_category
(
    category_id   bigserial    not null,
    category_name varchar(255) not null,
    primary key (category_id)
);

create table if not exists item
(
    item_id       bigserial    not null,
    amount        integer      not null,
    category_id   bigint       not null,
    cell_number   integer      not null,
    defect_amount integer      not null,
    name          varchar(255) not null,
    price         integer      not null,
    primary key (item_id),
    foreign key (category_id) references item_category (category_id)
);


create table if not exists delivery_list
(
    item_id     bigint not null,
    delivery_id bigint not null,
    amount      bigint not null,
    primary key (item_id, delivery_id),
    foreign key (delivery_id) references delivery (delivery_id),
    foreign key (item_id) references item (item_id)
);

create table if not exists customer
(
    customer_id bigserial    not null,
    email       varchar(255) not null unique,
    name        varchar(255) not null,
    second_name varchar(255) not null,
    primary key (customer_id)
);

create table if not exists orders
(
    order_id    bigserial    not null,
    full_price  integer      not null,
    order_date  timestamp(6) not null,
    customer_id bigint       not null,
    primary key (order_id),
    foreign key (customer_id) references customer (customer_id)
);

create table if not exists order_list
(
    order_id bigint not null,
    item_id  bigint not null,
    amount   bigint not null,
    primary key (order_id, item_id),
    foreign key (item_id) references item (item_id),
    foreign key (order_id) references orders (order_id)
);

create table if not exists cashier
(
    cashier_id  bigserial    not null,
    name        varchar(255) not null,
    second_name varchar(255) not null,
    primary key (cashier_id)
);


create table if not exists transaction_type
(
    type_id   bigserial    not null,
    type_name varchar(255) not null,
    primary key (type_id)
);

create table if not exists transaction
(
    transaction_id bigserial    not null,
    date           timestamp(6) not null,
    type_id        bigint       not null,
    cashier_id     bigint       not null,
    order_id       bigint       not null unique,
    primary key (transaction_id),
    foreign key (type_id) references transaction_type (type_id),
    foreign key (cashier_id) references cashier (cashier_id),
    foreign key (order_id) references orders (order_id)
);


