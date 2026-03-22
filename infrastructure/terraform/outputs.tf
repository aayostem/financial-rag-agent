output "vpc_id"                    { value = aws_vpc.main.id }
output "vpc_cidr_block"            { value = aws_vpc.main.cidr_block }
output "private_subnet_ids"        { value = aws_subnet.private[*].id }
output "public_subnet_ids"         { value = aws_subnet.public[*].id }
output "private_route_table_ids"   { value = aws_route_table.private[*].id }
output "public_route_table_id"     { value = aws_route_table.public.id }
output "vpc_endpoint_sg_id"        { value = aws_security_group.vpc_endpoints.id }
output "nat_gateway_public_ips"    { value = aws_eip.nat[*].public_ip }
